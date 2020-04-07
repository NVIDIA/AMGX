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

#pragma once

namespace amgx
{

template<int I>struct log2 {};
template<> struct log2<1> {static const int value = 0;};
template<> struct log2<2> {static const int value = 1;};
template<> struct log2<4> {static const int value = 2;};
template<> struct log2<8> {static const int value = 3;};
template<> struct log2<16> {static const int value = 4;};
template<> struct log2<32> {static const int value = 5;};
template<int I>struct pow2 {static const unsigned int value = 2 * pow2 < I - 1 >::value;};
template<> struct pow2<0> {static const unsigned int value = 1;};


template<int SUBWARP_LANES>
struct subwarp
{
    typedef subwarp<SUBWARP_LANES> self;
    //static const int LOG2_SUBWARP_LANES=??;

    static __device__ __forceinline__ unsigned int masked_ballot(const int pred)
    {
        unsigned int votes = utils::ballot(pred);
        return votes & self::mask();
    }
    static __device__ __forceinline__ int any(const int pred)
    {
        return __popc(self::masked_ballot(pred)) != 0;
    }
    static __device__ __forceinline__ int all(const int pred)
    {
        return __popc(self::masked_ballot(pred)) == SUBWARP_LANES;
    }
    static __device__ __forceinline__ unsigned int mask()
    {
        int lane_id_div_sl = utils::lane_id() / SUBWARP_LANES;
        unsigned int mask_tmp = pow2<SUBWARP_LANES>::value - 1; //__ballot( lane_id_div_sl == 0 );
        unsigned int ret = mask_tmp << (lane_id_div_sl * SUBWARP_LANES);
        //if(threadIdx.x/32 == 0 && blockIdx.x == 0) printf("%ud %ud %ud %ud\n",threadIdx.x,ret,SUBWARP_LANES,__popc(mask_tmp));
        return ret;
    }
    static __device__ __forceinline__ int subwarp_id()
    {
        return threadIdx.x / SUBWARP_LANES; //threadIdx/32 -> threadId/4
    }
    static __device__ __forceinline__ int sublane_id()
    {
        return threadIdx.x % SUBWARP_LANES;
    }
};

template<>
struct subwarp<32>
{
    static __device__ __forceinline__ unsigned int masked_ballot(const int pred)
    {
        return utils::ballot(pred);
    }
    static __device__ __forceinline__ int any(const int pred)
    {
        return utils::any(pred);
    }
    static __device__ __forceinline__ int all(const int pred)
    {
        return utils::all(pred);
    }
    static __device__ __forceinline__ unsigned int mask()
    {
        return 31;
    }
    static __device__ __forceinline__ int subwarp_id()
    {
        return utils::warp_id();
    }
    static __device__ __forceinline__ int sublane_id()
    {
        return utils::lane_id();
    }
};
template<>
struct subwarp<1>
{
    static __device__ __forceinline__ unsigned int masked_ballot(const int pred)
    {
        return pred;
    }
    static __device__ __forceinline__ int any(const int pred)
    {
        return pred;
    }
    static __device__ __forceinline__ int all(const int pred)
    {
        return pred;
    }
    static __device__ __forceinline__ int subwarp_id()
    {
        return threadIdx.x;
    }
    static __device__ __forceinline__ int sublane_id()
    {
        return 0;
    }
};

//deprecated
template<int BOX_BITS_>
struct flags_bf
{
    static const int BOX_BITS = BOX_BITS_;
    static const int COLOR_BITS = 64 - BOX_BITS;
    static const int MAX_COLORS = COLOR_BITS;
    static const unsigned long long BOXES_MASK = 1 << BOX_BITS - 1;
    static const unsigned long long COLORS_MASK = 1 << COLOR_BITS - 1;

    unsigned long long flags; //16 bit

    __forceinline__ __device__ int boxes()const
    {
        return (flags >> COLOR_BITS) & BOXES_MASK ;
    }
    __forceinline__ __device__ void set_boxes(const int boxes_count)
    {
        flags = (flags & COLORS_MASK) | (boxes_count << COLOR_BITS);
    }
    __forceinline__ __device__ void add_color(const int c)
    {
        int used_boxes = boxes();
        int min_color = used_boxes * COLOR_BITS;
        int c2 = c - min_color;

        if (c2 >= MAX_COLORS)
        {
            used_boxes = used_boxes + 1;
            set_boxes(used_boxes);
        }

        flags |= c2;
    }
    __forceinline__ __device__ bool color_available(const int c)
    {
        int used_boxes = boxes();
        int min_color = used_boxes * COLOR_BITS;
        int c2 = c - min_color;

        if (c2 <  0) { return false; }

        if (c2 >= MAX_COLORS) { return true; }

        return (1 << c2)&flags;
    }
    __forceinline__ __device__ int first_available_color()
    {
        int used_boxes = boxes();
        int min_color = used_boxes * COLOR_BITS;
        min_color = utils::bfind( ~flags  );
        return min_color + 64;
    }
};


struct used_color_structure_64_bit
{
    static const int COLORBOX_SIZE = 64;
    unsigned long long bitfield;

    struct aux_tmp
    {
        int box_id;
        __device__ __forceinline__ aux_tmp()
        {
            box_id = 0;
        }
    };

    __device__ __forceinline__ used_color_structure_64_bit()
    {
        bitfield = 0ull;
    }
    __device__ __forceinline__ int use_color(const int color, aux_tmp &aux)
    {
        bitfield |= (1ull << (64 - color));
        return (~bitfield) == 0ull;
    }
    template<int POSITION>
    __device__ __forceinline__ int available_color(aux_tmp &aux) const
    {
        unsigned long long bitfield_1 = bitfield;
        int color_i = 64 - utils::bfind( ~bitfield_1 );
#pragma unroll

        for (int i = 1; i < POSITION; i++)
        {
            bitfield_1 |= (1ull << (64 - color_i));
            color_i     = 64 - utils::bfind( ~bitfield_1 );
        }

        return color_i;
    }
    __device__ __forceinline__ int first_available_color(aux_tmp &aux) const
    {
        return 64 - utils::bfind( ~bitfield );
    }
    template<int CTA_SIZE, int WARP_SIZE>
    __device__ __forceinline__ void sync_subwarp(const int sublane_id)
    {
#if __CUDA_ARCH__ >= 300
#pragma unroll

        for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
        {
            int tmp_hi = __double2hiint( __longlong_as_double( bitfield ) );
            int tmp_lo = __double2loint( __longlong_as_double( bitfield ) );
            tmp_hi = utils::shfl_xor(tmp_hi, i, WARP_SIZE);
            tmp_lo = utils::shfl_xor(tmp_lo, i, WARP_SIZE);
            long long tmp = __double_as_longlong(__hiloint2double(tmp_hi, tmp_lo));
            bitfield |= tmp;
        }

#else
        __shared__ volatile long long s_used_colors[CTA_SIZE + WARP_SIZE / 2];
#pragma unroll

        for (int i = 1; i <= WARP_SIZE / 2; i *= 2)
        {
            s_used_colors[threadIdx.x] = bitfield;
            long long tmp = s_used_colors[threadIdx.x + i];

            if (sublane_id + i < WARP_SIZE) { bitfield |= tmp; }
        }

#endif
    }
    __device__ __forceinline__ void aggregate(const used_color_structure_64_bit &b)
    {
        bitfield = bitfield | b.bitfield;
    }
    __device__ __forceinline__ void set_boxid(const int j, aux_tmp &aux)
    {
        aux.box_id = j;
    }
};

template<int N_COLORBOXES_BITS>
struct used_color_structure_64_bit_colorbox
{
    static const int COLORBOX_SIZE = 64 - N_COLORBOXES_BITS;
    static const unsigned long long COLORBOX_MASK          = ((1ull << (COLORBOX_SIZE)) - 1);
    static const unsigned long long FULL_BOXES_NUMBER_MASK = ~COLORBOX_MASK;//(1ull<<(N_COLORBOXES_BITS)-1) << COLORBOX_SIZE;

    union
    {
        struct
        {
unsigned long long int used_colors :
            COLORBOX_SIZE;
unsigned long long int box_id :
            N_COLORBOXES_BITS;
        } s;
        unsigned long long int bitfield;
    } data;

    struct aux_tmp
    {

    };
    __device__ __forceinline__ void set_boxid(const int j, aux_tmp &aux)
    {
        data.s.box_id = j;
    }

    __device__ __forceinline__ used_color_structure_64_bit_colorbox()
    {
        data.bitfield = 0ull;
    }
    __device__ __forceinline__ unsigned long long get_full_boxes_number()const
    {
        return data.s.box_id;
    }
    __device__ __forceinline__ unsigned long long get_used_colors()const
    {
        return data.s.used_colors;
    }
    __device__ __forceinline__ void set_full_boxes_number(const unsigned long long &b)
    {
        data.s.box_id = b;
    }
    __device__ __forceinline__ void set_used_colors(const unsigned long long &b)
    {
        data.s.used_colors = b;
    }
    __device__ __forceinline__ int box_basecolor() const
    {
        return (get_full_boxes_number()) * COLORBOX_SIZE;
    }
    __device__ __forceinline__ int max_color_in_box() const
    {
        return (get_full_boxes_number() + 1) * COLORBOX_SIZE;
    }
    __device__ __forceinline__ int use_color(const int color_, aux_tmp &aux)
    {
        int color = color_ - 1;
        int color_in_box = color - box_basecolor();
        int max_in_box = max_color_in_box();

        if (color_in_box >= max_in_box)
        {
            set_full_boxes_number( color / COLORBOX_SIZE );
            set_used_colors( 0ull );
            return 1;
        }

        set_used_colors( get_used_colors() | (1ull << (COLORBOX_SIZE - color_in_box - 1)) );
        unsigned long long available_colors = ~(get_used_colors() | FULL_BOXES_NUMBER_MASK);

        if (available_colors == 0ull) //finished current box: all zeros
        {
            set_full_boxes_number((get_full_boxes_number()) + 1);
            set_used_colors(0ull);
            return 1;
        }

        return 0;
    }
    __device__ __forceinline__ int first_available_color(aux_tmp &aux) const
    {
        unsigned long long available_colors = ~(get_used_colors() | FULL_BOXES_NUMBER_MASK);

        if (available_colors == 0ull) { return 0; }

        int color_in_box = COLORBOX_SIZE - utils::bfind( available_colors );
        int ret = box_basecolor() + color_in_box;
        return ret;
    }

    __device__ __forceinline__ void aggregate(const used_color_structure_64_bit_colorbox<N_COLORBOXES_BITS> &b)
    {
        if (b.data.s.box_id > data.s.box_id) //b wins
        {
            data.bitfield = b.data.bitfield;
        }
        else if (b.data.s.box_id == data.s.box_id)
        {
            data.s.used_colors |= b.data.s.used_colors;
        }
        else
        {
            //do nothing: b loses
        }
    }
    template<int CTA_SIZE, int WARP_SIZE>
    __device__ __forceinline__ void sync_subwarp(const int sublane_id)
    {
        used_color_structure_64_bit_colorbox<N_COLORBOXES_BITS> tmps;
#if __CUDA_ARCH__ >= 300
#pragma unroll

        for (int i = WARP_SIZE / 2; i >= 1; i /= 2)
        {
            int tmp_hi = __double2hiint( __longlong_as_double( data.bitfield ) );
            int tmp_lo = __double2loint( __longlong_as_double( data.bitfield ) );
            tmp_hi = utils::shfl_xor(tmp_hi, i, WARP_SIZE);
            tmp_lo = utils::shfl_xor(tmp_lo, i, WARP_SIZE);
            long long tmp = __double_as_longlong(__hiloint2double(tmp_hi, tmp_lo));
            tmps.data.bitfield = tmp;  //get neighbors bitfield
            aggregate(tmps);
        }

#else
        __shared__ volatile unsigned long long s_used_colors[CTA_SIZE + WARP_SIZE / 2];
#pragma unroll

        for (int i = 1; i <= WARP_SIZE / 2; i *= 2)
        {
            s_used_colors[threadIdx.x] = data.bitfield;
            unsigned long long tmp = s_used_colors[threadIdx.x + i];
            tmps.data.bitfield = tmp; //get neighbors bitfield

            if (sublane_id + i < WARP_SIZE) { aggregate(tmps); }
        }

#endif
    }
};

int eliminate_null_colors(int num_rows, int max_color, int *row_colors); //impl in greedy_recolor.cu
int reorder_colors_by_frequency(int num_rows, int max_color, int *row_colors); //impl in greedy_recolor.cu
void coloring_histogram(int *out_hist, int num_rows, int max_color, int *row_colors); //impl in greedy_recolor.cu
int reverse_colors(int num_rows, int max_color, int *row_colors); //impl in greedy_recolor.cu


}
