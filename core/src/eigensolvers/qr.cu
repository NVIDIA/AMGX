/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <eigensolvers/eigensolver.h>
#include <cublas_v2.h>
#include <matrix.h>
#include <eigensolvers/qr.h>
#include <amgx_lapack.h>
#include <amgx_cublas.h>
#include <stack>
#include <queue>
#include <fstream>
#include <blas.h>
#include <norm.h>

namespace amgx
{

namespace
{

template <typename T>
__global__
void kernel_clear_lower_triangular(T *V, int dim, int lda)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < dim)
    {
        for (int c = 0; c < r; ++c)
        {
            V[c * lda + r] = 0.;
        }
    }
}

template <typename Vector>
void clear_lower_triangular(Vector &V)
{
    int num_threads = 128;
    int max_grid_size = 4096;
    int num_rows = V.get_num_rows();
    int num_blocks = std::min(max_grid_size, (num_rows + num_threads - 1) / num_rows);
    kernel_clear_lower_triangular <<< num_blocks, num_threads>>>(V.raw(), num_rows, V.get_lda());
    cudaCheckError();
}


struct TreeNode
{
    TreeNode(int v)
        : left(0), right(0), parent(0), value(v)
    {
    }

    bool is_leaf() const
    {
        return left == right;
    }

    bool is_root() const
    {
        return !parent;
    }

    TreeNode *left;
    TreeNode *right;
    TreeNode *parent;
    int value;
};

// Recursive function to print a reduction tree to an ofstream in dot
// format.
void print_tree_rec(TreeNode *n, std::ofstream &f)
{
    TreeNode *left = n->left;
    TreeNode *right = n->right;
    TreeNode *parent = n->parent;
    // We use the address of the nodes as labels in dot format to
    // avoid having to renumber the nodes.
    size_t address = reinterpret_cast<size_t>(n);
    size_t parent_address = reinterpret_cast<size_t>(parent);
    size_t left_address = reinterpret_cast<size_t>(left);
    size_t right_address = reinterpret_cast<size_t>(right);
    f << address << "[label=" << n->value << "];" << std::endl;

    if (parent)
    {
        f << address << " -> " << parent_address << std::endl;
    }

    if (left)
    {
        f << address << " -> " << left_address << std::endl;
        print_tree_rec(left, f);
    }

    if (right)
    {
        f << address << " -> " << right_address << std::endl;
        print_tree_rec(right, f);
    }
}

#ifdef DEBUG
// Debug function: print a reduction tree to a file in dot format.
void print_tree(TreeNode *root, const char *filename)
{
    std::ofstream f;
    f.open(filename);
    f << "digraph A {" << std::endl;
    print_tree_rec(root, f);
    f << "}" << std::endl;
}

// Create a degenerate tree of size N, this is not an efficient
// reduction tree for a parallel operation (rank 0 is involved in all
// operations) but it is used for testing.
// Example with N == 4:
//       0
//      / \
//     0   \
//    / \   \
//   0   \   \
//  / \   \   \
// 0   1   2   3
// The label on each node is the active process for this operation.
TreeNode *build_degenerate_tree(int N)
{
    std::vector<TreeNode *> leaves;

    // Allocate one leaf node for each rank.
    for (int i = 0; i < N; ++i)
    {
        leaves.push_back(new TreeNode(i));
    }

    TreeNode *previous_node = leaves[0];

    // For each leaf except the first one, create the new root of the
    // tree labeled with 0.
    // The left child of this node is the previous root and the right
    // child is the current leaf.
    for (int i = 0; i < N - 1; ++i)
    {
        TreeNode *left = previous_node;
        TreeNode *right = leaves[i + 1];
        TreeNode *ancestor = new TreeNode(previous_node->value);
        left->parent = ancestor;
        right->parent = ancestor;
        ancestor->left = left;
        ancestor->right = right;
        previous_node = ancestor;
    }

    TreeNode *root = previous_node;
    return root;
}

#endif
// Create a full tree of size N, N must be a power of 2.
// For instance if N == 4:
//        0
//       / \
//      /   \
//     0     2
//    / \   / \
//   0   1 2   3
// The label on each node is the active process for this operation.
TreeNode *build_full_tree(int N)
{
    // Nodes of the previous (deepest) level of the tree.
    std::vector<TreeNode *> previous_level;
    std::vector<TreeNode *> current_level;

    // Allocate N leaves nodes, one for each process.
    for (int i = 0; i < N; ++i)
    {
        previous_level.push_back(new TreeNode(i));
    }

    // Level i of the tree will have 2^i nodes, we construct the tree
    // from leaves to root. At each level we allocate new nodes and
    // connect these nodes to the previous level.
    for (int stride = 2; stride <= N; stride <<= 1)
    {
        current_level.clear();
        int children_index = 0;

        for (int j = 0; j < N; j += stride)
        {
            TreeNode *left = previous_level[children_index++];
            TreeNode *right = previous_level[children_index++];
            TreeNode *ancestor = new TreeNode(j);
            left->parent = ancestor;
            right->parent = ancestor;
            ancestor->left = left;
            ancestor->right = right;
            current_level.push_back(ancestor);
        }

        previous_level = current_level;
    }

    TreeNode *root = current_level.front();
    return root;
}

void delete_tree(TreeNode *root)
{
    if (!root)
    {
        return;
    }

    delete_tree(root->left);
    delete_tree(root->right);
    delete root;
}

int previous_power_2(int v)
{
    unsigned p = 1;

    while (p <= v)
    {
        p <<= 1;
    }

    return p >> 1;
}

// Build a reduction tree of size N. We build a full tree of size M,
// the nearest lower power of 2 and we add remaining nodes in a
// similar process as build_degenerate tree: rank 0 is involved in all
// operations.

TreeNode *build_reduction_tree(int N)
{
    int nearest_pow2 = previous_power_2(N);
    TreeNode *root = build_full_tree(nearest_pow2);

    if (nearest_pow2 == N)
    {
        return root;
    }

    // Number of ranks is not a power of two, create additional nodes for the remaining ranks.
    TreeNode *previous = root;
    int root_value = root->value;

    for (int i = nearest_pow2; i < N; ++i)
    {
        TreeNode *left = previous;
        TreeNode *right = new TreeNode(i);
        TreeNode *ancestor = new TreeNode(root_value);
        left->parent = ancestor;
        right->parent = ancestor;
        ancestor->left = left;
        ancestor->right = right;
        previous = ancestor;
    }

    return previous;
}

// Given a reduction tree, we have to compute for a given rank what
// are the actions to perform, i.e. we need to find the list of nodes
// in the tree whose value is the current rank.
// For the up-sweep order, we need the list of these nodes ordered
// from bottom to top (root). This order is given by performing a
// breadth-first traversal of the tree and pushing encountered nodes
// to a stack.
void compute_up_sweep_order(TreeNode *root, int v, std::stack<TreeNode *> &order)
{
    std::queue<TreeNode *> to_visit;
    to_visit.push(root);

    while (!to_visit.empty())
    {
        TreeNode *current = to_visit.front();
        to_visit.pop();

        if (current->value == v)
        {
            order.push(current);
        }

        if (current->left)
        {
            to_visit.push(current->left);
        }

        if (current->right)
        {
            to_visit.push(current->right);
        }
    }
}

template <typename TConfig>
void
vstack(Vector<TConfig> &dst, const Vector<TConfig> &top, const Vector<TConfig> &bottom)
{
    typedef typename Vector<TConfig>::value_type value_type;
    dst.resize((top.get_num_rows() + bottom.get_num_rows()) * top.get_num_cols());
    dst.set_num_rows(top.get_num_rows() + bottom.get_num_rows());
    dst.set_num_cols(top.get_num_cols());
    dst.set_lda(top.get_num_rows() + bottom.get_num_rows());
    int dpitch = dst.get_lda() * sizeof(value_type);
    int spitch = top.get_lda() * sizeof(value_type);
    int width = top.get_num_cols() * sizeof(value_type);
    int height = top.get_num_rows();
    cudaMemcpy2D(dst.raw(), dpitch, top.raw(), spitch, width, height,
                 cudaMemcpyDeviceToDevice);
    int offset = top.get_num_cols();
    cudaMemcpy2D(dst.raw() + top.get_num_cols(), dpitch, bottom.raw(), spitch, width, height,
                 cudaMemcpyDeviceToDevice);
}

template <typename Vector>
void vslice(const Vector &src, Vector &dst, int start, int end)
{
    typedef typename Vector::value_type value_type;
    int cols = src.get_num_cols();
    int dst_rows = end - start;
    dst.resize(dst_rows * cols);
    dst.set_num_rows(dst_rows);
    dst.set_num_cols(cols);
    dst.set_lda(dst_rows);
    int dpitch = dst.get_lda() * sizeof(value_type);
    int spitch = src.get_lda() * sizeof(value_type);
    int width = dst.get_num_cols() * sizeof(value_type);
    int height = dst.get_num_rows();
    cudaMemcpy2D(dst.raw(), dpitch, src.raw() + start, spitch, width, height,
                 cudaMemcpyDeviceToDevice);
}

} // end anonymous namespace.

template <typename TConfig>
HouseholderQR<TConfig>::HouseholderQR(TMatrix &A)
    : m_A(A), m_use_R_inverse(true)
{
}

template <typename TConfig>
void HouseholderQR<TConfig>::QR(TVector &V)
{
    int rows = V.get_num_rows();
    int cols = V.get_num_cols();
    int lda = V.get_lda();
    Lapack<TConfig>::geqrf(V, m_tau, m_work);
    Lapack<TConfig>::orgqr(V, m_tau, m_work);
}

template <typename TConfig>
void HouseholderQR<TConfig>::QR(TVector &V, TVector &R)
{
    int rows = V.get_num_rows();
    int cols = V.get_num_cols();
    int lda = V.get_lda();
    R.resize(cols * cols);
    R.set_num_rows(cols);
    R.set_num_cols(cols);
    R.set_lda(cols);
    Lapack<TConfig>::geqrf(V, m_tau, m_work);
    vslice(V, R, 0, cols);
    clear_lower_triangular(R);
    Lapack<TConfig>::orgqr(V, m_tau, m_work);
}


// Send a vector to rank 'destination'.
// If 'destination' is the current rank, we don't perform any call to
// MPI and instead save the vector in a stack.
template <typename TConfig>
void
HouseholderQR<TConfig>::send_vector(TVector &V, int destination)
{
    TMatrix &A = m_A;
    int current_rank = A.manager->global_id();

    if (destination == current_rank)
    {
        m_local_comms_stack.push(V);
    }
    else
    {
        A.manager->getComms()->send_vector(V, destination, 0);
    }
}

// Receive a vector from rank 'source'.
// If 'source' is the current rank, we just pop the vector from our
// local stack.
template <typename TConfig>
void
HouseholderQR<TConfig>::receive_vector(TVector &V, int source)
{
    TMatrix &A = m_A;
    int current_rank = A.manager->global_id();

    if (source == current_rank)
    {
        V = m_local_comms_stack.top();
        m_local_comms_stack.pop();
    }
    else
    {
        A.manager->getComms()->recv_vector(V, source, 0);
    }
}

// Instead of performing the down-sweep phase of the TSQR algorithm to
// get the Q matrix, we can compute Q = A * R ^ -1. This alternative
// use less communications.
template <typename TConfig>
void HouseholderQR<TConfig>::inverse_phase(TVector &V, TVector &R, int root)
{
    TMatrix &A = m_A;
    int num_partitions = A.manager->getComms()->get_num_partitions();
    int current_partition = A.manager->global_id();

    if (current_partition == root)
    {
        for (int i = 0; i < num_partitions; ++i)
        {
            send_vector(R, i);
        }
    }

    receive_vector(R, root);
    Vector_h h_R = R;
    Lapack<TConfig_h>::trtri(h_R);
    R = h_R;
    // Q = V * R^-1
    Cublas::gemm(1, V, R, 0, V);
}

template <typename TConfig>
void HouseholderQR<TConfig>::QR_decomposition(TVector &V)
{
    int rows = V.get_num_rows();
    int cols = V.get_num_cols();
    int lda = V.get_lda();
    TMatrix &A = m_A;

    if (cols == 1)
    {
        ValueTypeVec norm = get_norm(A, V, L2);
        scal(V, 1 / norm, 0, rows);
        return;
    }

    m_tau.resize(cols);
    m_work.resize(rows * cols);

    if (A.is_matrix_singleGPU())
    {
        QR(V);
        return;
    }

    // Multi-GPU path.
    int num_partitions = A.manager->getComms()->get_num_partitions();
    int current_partition = A.manager->global_id();
    // Stack of the Q matrices compute during reduction.
    std::stack<TVector> Q_stack;
    TVector B1(cols * cols);
    B1.set_num_rows(cols);
    B1.set_num_cols(cols);
    B1.set_lda(cols);
    TVector B2(cols * cols);
    B2.set_num_rows(cols);
    B2.set_num_cols(cols);
    B2.set_lda(cols);
    TVector B;
    TVector R;
    TreeNode *reduce_tree = build_reduction_tree(num_partitions);
    // Debug: uncomment to print reduction tree.
    /*
        if (current_partition == 0)
            print_tree(reduce_tree, "tree.dot");
    */
    std::stack<TreeNode *> up_sweep_order;
    compute_up_sweep_order(reduce_tree, current_partition, up_sweep_order);
    std::stack<TreeNode *> down_sweep_order;
    std::stack<TreeNode *> order = up_sweep_order;

    while (!order.empty())
    {
        down_sweep_order.push(order.top());
        order.pop();
    }

    // Up-sweep phase:
    // - At each leaf node: compute a QR on its part of the vector then send the R matrix to its parent.
    // - At each non-leaf node: receives a R matrix from both
    //   children, compute a QR and send the result R to the parent.
    //   Q matrices are saved in a stack.
    while (!up_sweep_order.empty())
    {
        TreeNode *current = up_sweep_order.top();
        up_sweep_order.pop();
        TreeNode *left_child = current->left;
        TreeNode *right_child = current->right;
        TreeNode *parent = current->parent;

        if (current->is_leaf())
        {
            B = V;
        }
        else
        {
            // Receive R matrices from children and stack the two one above the other.
            receive_vector(B1, left_child->value);
            receive_vector(B2, right_child->value);
            vstack(B, B1, B2);
        }

        // costly for first iteration since we have to copy the vector to the stack.
        QR(B, R);
        Q_stack.push(B);

        if (!current->is_root())
        {
            send_vector(R, parent->value);
        }
    }

    if (m_use_R_inverse)
    {
        int reduce_root = reduce_tree->value;
        delete_tree(reduce_tree);
        inverse_phase(V, R, reduce_root);
        return;
    }

    // Down-sweep phase:
    // - At each node: receive a matrix from your parent (root of the
    //   tree uses the identity matrix) and multiply this matrix with
    //   the saved Q matrix for this node (computed during the
    //   up-sweep phase).
    //   Split this vector vertically and send the first half to your
    //   left child and the second half to your right child.
    //   Leaves nodes do not send anything, after multiplication the
    //   result is the local part of the global Q matrix.
    while (!down_sweep_order.empty())
    {
        TreeNode *current = down_sweep_order.top();
        down_sweep_order.pop();
        TreeNode *left_child = current->left;
        TreeNode *right_child = current->right;
        TreeNode *parent = current->parent;
        TVector Q = Q_stack.top();
        Q_stack.pop();

        if (current->is_root())
        {
            B = Q;
        }
        else
        {
            // If leaf, write directly to output vector.
            TVector &output = current->is_leaf() ? V : B;
            receive_vector(B1, parent->value);
            Cublas::gemm(1, Q, B1, 0, output);
        }

        if (current->is_leaf())
        {
            continue;
        }

        int rows = B.get_num_rows();
        vslice(B, B1, 0, rows / 2);
        vslice(B, B2, rows / 2, rows);
        send_vector(B1, left_child->value);
        send_vector(B2, right_child->value);
    }

    delete_tree(reduce_tree);
}

#define AMGX_CASE_LINE(CASE) template class HouseholderQR<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}

