/*
 *  Copyright 2011 The Regents of the University of California
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>

namespace blas = cusp::blas;
namespace cusp
{
  namespace krylov
  {    
    template <typename ValueType> 
    void ApplyPlaneRotation(ValueType& dx,
			    ValueType& dy,
			    ValueType& cs,
			    ValueType& sn)
    {
      ValueType temp = cs * dx + sn *dy;
      dy = -sn*dx+cs*dy;
      dx = temp;
    }

    template <typename ValueType>
    void GeneratePlaneRotation(ValueType& dx,
			       ValueType& dy,
			       ValueType& cs,
			       ValueType& sn)
    {
      if(dy == ValueType(0.0)){
	cs = 1.0;
	sn = 0.0;
      }else if (abs(dy) > abs(dx)) {
	ValueType tmp = dx / dy;
	sn = ValueType(1.0) / sqrt(ValueType(1.0) + tmp*tmp);
	cs = tmp*sn;            
      }else {
	ValueType tmp = dy / dx;
	cs = ValueType(1.0) / sqrt(ValueType(1.0) + tmp*tmp);
	sn = tmp*cs;
      }
    }

    template <class LinearOperator,typename ValueType> 
    void PlaneRotation(LinearOperator& H,
		       ValueType& cs,
		       ValueType& sn,
		       ValueType& s,
		       int i)
    {
      for (int k = 0; k < i; k++){
	ApplyPlaneRotation(H(k,i), H(k+1,i), cs[k], sn[k]);
      }
      GeneratePlaneRotation(H(i,i), H(i+1,i), cs[i], sn[i]);
      ApplyPlaneRotation(H(i,i), H(i+1,i), cs[i], sn[i]);
      ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i]);
    }

    template <class LinearOperator,
	      class Vector>
    void gmres(LinearOperator& A,
	       Vector& x,
	       Vector& b,
	       const size_t restart)
    {
      typedef typename LinearOperator::value_type   ValueType;
      cusp::default_monitor<ValueType> monitor(b);
      cusp::krylov::gmres(A, x, b, restart, monitor);
    }

    template <class LinearOperator,
	      class Vector,
	      class Monitor>
    void gmres(LinearOperator& A,
	       Vector& x,
	       Vector& b,
	       const size_t restart,
	       Monitor& monitor)
    {
      typedef typename LinearOperator::value_type   ValueType;
      typedef typename LinearOperator::memory_space MemorySpace;
      cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);
      cusp::krylov::gmres(A, x, b, restart, monitor, M);
    }
    
    template <class LinearOperator,
	      class Vector,
	      class Monitor,
	      class Preconditioner>
    void gmres(LinearOperator& A,
	       Vector& x,
	       Vector& b,
	       const size_t restart,
	       Monitor& monitor,
	       Preconditioner& M)
    {
      typedef typename LinearOperator::value_type   ValueType;
      typedef typename LinearOperator::memory_space MemorySpace;
      typedef typename norm_type<ValueType>::type NormType;
      assert(A.num_rows == A.num_cols);        // sanity check
      const size_t N = A.num_rows;
      const int R = restart;
      int i, j, k;
      NormType beta = 0;
      NormType resid0 = 0;
      cusp::array1d<NormType,cusp::host_memory> rel_resid(1);
      //allocate workspace
      cusp::array1d<ValueType,MemorySpace> w(N);
      cusp::array1d<ValueType,MemorySpace> V0(N); //Arnoldi matrix pos 0
      cusp::array2d<ValueType,MemorySpace,cusp::column_major> V(N,R+1,ValueType(0.0)); //Arnoldi matrix
      //duplicate copy of s on GPU
      cusp::array1d<ValueType,MemorySpace> sDev(R+1);
      //HOST WORKSPACE
      cusp::array2d<ValueType,cusp::host_memory,cusp::column_major> H(R+1, R); //Hessenberg matrix
      cusp::array1d<ValueType,cusp::host_memory> s(R+1);
      cusp::array1d<ValueType,cusp::host_memory> cs(R);
      cusp::array1d<ValueType,cusp::host_memory> sn(R);
      NormType b_norm = blas::nrm2(b);
      
      do{
	// compute initial residual and its norm //
	cusp::multiply(A, x, w);                     // V(0) = A*x        //
	blas::axpy(b,w,ValueType(-1));               // V(0) = V(0) - b   //
	cusp::multiply(M,w,w);                       // V(0) = M*V(0)     //
	beta = blas::nrm2(w);                        // beta = norm(V(0)) //
	blas::scal(w, ValueType(-1.0/beta));         // V(0) = -V(0)/beta //
	blas::copy(w,V.column(0));
	// save very first residual norm //
	if (monitor.iteration_count()== 0){
	  //resid0 = beta;
	  cusp::multiply(M,b,V0);
	  resid0 = blas::nrm2(V0)/b_norm;
	}
	//s = 0 //
	blas::fill(s,ValueType(0.0));
	s[0] = beta;
	i = -1;
	
	do{
	  ++i;
	  ++monitor;
	  
	  //apply preconditioner
	  //can't pass in ref to column in V so need to use copy (w)
	  cusp::multiply(A,w,V0);
	  //V(i+1) = A*w = M*A*V(i)    //
	  cusp::multiply(M,V0,w);
	  
	  for (k = 0; k <= i; k++){
	    //  H(k,i) = <V(i+1),V(k)>    //
	    H(k, i) = blas::dotc(w, V.column(k));
	    // V(i+1) -= H(k, i) * V(k)  //
	    blas::axpy(V.column(k),w,-H(k,i));
	  }
	  
	  H(i+1,i) = blas::nrm2(w);   
	  // V(i+1) = V(i+1) / H(i+1, i) //
	  blas::scal(w,ValueType(1.0)/H(i+1,i));
	  blas::copy(w,V.column(i+1));
	  
	  PlaneRotation(H,cs,sn,s,i);
	  
	  rel_resid[0] = abs(s[i+1]) / resid0 + monitor.absolute_tolerance();
	  
	  //check convergence condition
	  //if (rel_resid < monitor.relative_tolerance())
	  if (monitor.finished(rel_resid)){
	    break;
	  }
	}while (i+1 < R && monitor.iteration_count()+1 <= monitor.iteration_limit());
	

	// solve upper triangular system in place //
	for (j = i; j >= 0; j--){
	  s[j] /= H(j,j);
	  //S(0:j) = s(0:j) - s[j] H(0:j,j)
	  for (k = j-1; k >= 0; k--){
	    s[k] -= H(k,j) * s[j];
	  }
	}
	
	// update the solution //
	
	//copy s to gpu 
	blas::copy(s,sDev);
	// x= V(1:N,0:i)*s(0:i)+x //
	for (j = 0; j <= i; j++){
	  // x = x + s[j] * V(j) //
	  blas::axpy(V.column(j),x,s[j]);
	}
      } while (rel_resid[0] >= monitor.tolerance() && 
	       monitor.iteration_count()+1 <= monitor.iteration_limit());
    }
  } // end namespace krylov
} // end namespace cusp
