/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <map>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <sstream>


namespace amgx
{
#if defined ( _MSC_VER ) && _MSC_VER <= 1700
//MSVC c++11 workaround 
template<typename T>
struct MemManager
{
        typedef std::map < T *, std::shared_ptr<T> > MemPoolMap;

        MemPoolMap &get_pool(void)
        {
            return pool_;
        }

        //have pool_ take ownership of shared pointer
        //(more exception safe than raw pointer version,
        //because it invites in-place "new" operator invocation
        //inside shared_ptr constructor)
        //
        void manage_ptr(std::shared_ptr<T> pX)
        {
            pool_.insert(std::make_pair(pX.get(), pX));
        }

        std::shared_ptr<T> manage_ptr(T *prawX)
        {
            std::shared_ptr<T> pX(prawX);
            pool_.insert(std::make_pair(pX.get(), pX));
            return pX;
        }

        //return an alias shared pointer using
        //the raw pointer as key
        //
        std::shared_ptr<T> get_pointer(T *prawX)
        {
            MemPoolMap &mX = pool_ ;
            auto pos = mX.find(prawX);

            if (pos != mX.end())
            {
                return std::shared_ptr<T>(pos->second, prawX);//shared_ptr aliasing constructor:
                //new participant in the
                //ownership of prawX;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: " << prawX << " is not pool-memory managed.";
                throw std::runtime_error(ss.str());
            }
        }

        //(the compiler can deduce the Args from arglist, up to a max of 5 args)
        //
        template<typename T1, typename T2, typename T3, typename T4, typename T5>
        std::shared_ptr<T> allocate(const T1 &arg1, const T2 &arg2, const T3 &arg3, const T4 &arg4, const T5 &arg5)
        {
            return manage_ptr(new T(arg1, arg2, arg3, arg4, arg5));
        }

        //(the compiler can deduce the Args from arglist, up to a max of 5 args)
        //
        template<typename T1, typename T2, typename T3, typename T4>
        std::shared_ptr<T> allocate(const T1 &arg1, const T2 &arg2, const T3 &arg3, const T4 &arg4)
        {
            return manage_ptr(new T(arg1, arg2, arg3, arg4));
        }

        //(the compiler can deduce the Args from arglist, up to a max of 5 args)
        //
        template<typename T1, typename T2, typename T3>
        std::shared_ptr<T> allocate(const T1 &arg1, const T2 &arg2, const T3 &arg3)
        {
            return manage_ptr(new T(arg1, arg2, arg3));
        }

        //(the compiler can deduce the Args from arglist, up to a max of 5 args)
        //
        template<typename T1, typename T2>
        std::shared_ptr<T> allocate(const T1 &arg1, const T2 &arg2)
        {
            return manage_ptr(new T(arg1, arg2));
        }

        //(the compiler can deduce the Args from arglist, up to a max of 5 args)
        //
        template<typename T1>
        std::shared_ptr<T> allocate(const T1 &arg1)
        {
            return manage_ptr(new T(arg1));
        }

        //(the compiler can deduce the Args from arglist, up to a max of 5 args)
        //
        std::shared_ptr<T> allocate(void)
        {
            return manage_ptr(new T);
        }

        //remove entry keyed by prawX
        //
        bool free(T *prawX)
        {
            return (pool_.erase(prawX) > 0);
        }

        size_t get_use_count(T *prawX)
        {
            MemPoolMap &mX = pool_;
            auto pos = mX.find(prawX);

            if (pos != mX.end())
            {
                return pos->second.use_count();
            }
            else
            {
                return 0;
            }
        }


    private:
        MemPoolMap pool_;
};

//singleton memory manager of a set of types
//
template<typename T>
MemManager<T> &get_mem_manager(void)
{
    static MemManager<T> man_;
    return man_;
}

#else // other compilers

template<typename T>
using MemPoolMap = std::map < T *, std::shared_ptr<T> > ;

//variadic template manager to hold
//a variadic set of memory pool maps
//into a tuple
//
template<typename... Types>
struct MemManager
{
        std::tuple<MemPoolMap<Types>...> &get_pools(void)
        {
            return pools_;
        }

        //assuming pools_ holds a type X at position tpl_pos,
        //have pool_ take ownership of raw pointer & return
        //corresponding shared_ptr
        //
        template<typename X, size_t tpl_pos>
        std::shared_ptr<X> manage_ptr(X *prawX)
        {
            MemPoolMap<X> &mX = std::get<tpl_pos>(pools_);
            std::shared_ptr<X> pX(prawX);
            mX.insert(std::make_pair(pX.get(), pX));
            return pX;
        }

        //assuming pools_ holds a type X at position tpl_pos,
        //have pool_ take ownership of shared pointer
        //(more exception safe than raw pointer version,
        //because it invites in-place "new" operator invocation
        //inside shared_ptr constructor)
        //
        template<typename X, size_t tpl_pos>
        void manage_ptr(std::shared_ptr<X> pX)
        {
            MemPoolMap<X> &mX = std::get<tpl_pos>(pools_);
            mX.insert(std::make_pair(pX.get(), pX));
        }

        //assuming pools_ holds a type X at position tpl_pos,
        //return an alias shared pointer using
        //the raw pointer as key
        //for the given type X managed at position tpl_pos
        //in the pool_
        //
        template<typename X, size_t tpl_pos>
        std::shared_ptr<X> get_pointer(X *prawX)
        {
            MemPoolMap<X> &mX = std::get<tpl_pos>(pools_);
            auto pos = mX.find(prawX);

            if (pos != mX.end())
            {
                return std::shared_ptr<X>(pos->second, prawX);//aliasing constructor:
                //new participant in the
                //ownership of prawX;
            }
            else
            {
                std::stringstream ss;
                ss << "ERROR: " << prawX << " is not pool-memory managed.";
                throw std::runtime_error(ss.str());
            }
        }

        //assuming pools_ holds a type X at position tpl_pos,
        //allocate a new X using variadic args constructor
        //(the compiler can deduce the Args from arglist,
        // only first 2 template arguments need to be specified)
        //
        template<typename X, size_t tpl_pos, typename...Args>
        std::shared_ptr<X> allocate(Args...args)
        {
            return manage_ptr<X, tpl_pos>(new X(args...));
        }

        //convenience allocator when there's only one managed type
        //
        template<typename X, typename...Args>
        std::shared_ptr<X> allocate(Args...args)
        {
            return allocate<X, 0>(args...);
        }

        //assuming pools_ holds a type X at position tpl_pos,
        //remove entry keyed by prawX
        //
        template<typename X, size_t tpl_pos>
        bool free(X *prawX)
        {
            MemPoolMap<X> &mX = std::get<tpl_pos>(pools_);
            return (mX.erase(prawX) > 0);
        }

        //convenience deallocator when there's only one managed type
        //
        template<typename X>
        bool free(X *prawX)
        {
            MemPoolMap<X> &mX = std::get<0>(pools_);
            return (mX.erase(prawX) > 0);
        }


        template<typename X, size_t tpl_pos>
        size_t get_use_count(X *prawX)
        {
            MemPoolMap<X> &mX = std::get<tpl_pos>(pools_);
            auto pos = mX.find(prawX);

            if (pos != mX.end())
            {
                return pos->second.use_count();
            }
            else
            {
                return 0;
            }
        }


    private:
        //unpacking of variadic template parameter pack:
        //suffix ... operator unpacks the pack
        //unpacking happens before tuple-ization
        //hence the tuple becomes a tuple of the pack
        //
        std::tuple<MemPoolMap<Types>...> pools_;
};

//singleton memory manager of a set of types
//
template<typename...Types>
MemManager<Types...> &get_mem_manager(void)
{
    static MemManager<Types...> man_;
    return man_;
}
#endif //#if defined ( _MSC_VER ) ...


namespace  //unnamed
{

//Managed Types:
//AMG_Configuration
//Matrix<TConfig>
//Vector<TConfig>
//AMG_Solver<TConfig>

typedef MemManager<AMG_Configuration> ConfigMemManager;

}//unnamed namespace

namespace //unnamed
{
//this does the bookkeeping of modes;
//it must be parameterized by the Handler type, only;
//
//this is necessary because several handled (Letter)
//types are parameterized by the mode

template<typename Handler>
std::map < Handler, int > &get_mode_bookkeeper(void)
{
    static std::map < Handler, int > mode_keepper_;
    return mode_keepper_;
}

}

//This is a Wrapper where the C-type (Envelope),
//typically some AMG_xxx_handle,
//and the underlying C++ type (Letter) are clearly
//identified
//
template <typename Envelope, typename Letter>
struct CWrapHandle
{
        CWrapHandle() : hdl_(0)
        {
        }
#if defined ( _MSC_VER ) && _MSC_VER <= 1700
//MSVC c++11 workaround 
//replace variadic templates with list of up to 5 args
        explicit CWrapHandle(Letter & /*not used*/)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter);
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2, typename T3, typename T4, typename T5>
        CWrapHandle(Letter & /*not used*/, T1 &arg1, T2 &arg2, T3 &arg3, T4 &arg4, T5 &arg5)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2, arg3, arg4, arg5));
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2, typename T3, typename T4>
        CWrapHandle(Letter & /*not used*/, T1 &arg1, T2 &arg2, T3 &arg3, T4 &arg4)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2, arg3, arg4));
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2, typename T3>
        CWrapHandle(Letter & /*not used*/, T1 &arg1, T2 &arg2, T3 &arg3)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2, arg3));
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2>
        CWrapHandle(Letter & /*not used*/, T1 &arg1, T2 &arg2)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2));
            hdl_ = (Envelope)this;
        }

        template<typename T1>
        CWrapHandle(Letter & /*not used*/, T1 &arg1)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1));
            hdl_ = (Envelope)this;
        }


        explicit CWrapHandle(Letter * /*not used*/)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter);
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2, typename T3, typename T4, typename T5>
        CWrapHandle(Letter * /*not used*/, T1 &arg1, T2 &arg2, T3 &arg3, T4 &arg4, T5 &arg5)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2, arg3, arg4, arg5));
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2, typename T3, typename T4>
        CWrapHandle(Letter * /*not used*/, T1 &arg1, T2 &arg2, T3 &arg3, T4 &arg4)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2, arg3, arg4));
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2, typename T3>
        CWrapHandle(Letter * /*not used*/, T1 &arg1, T2 &arg2, T3 &arg3)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2, arg3));
            hdl_ = (Envelope)this;
        }

        template<typename T1, typename T2>
        CWrapHandle(Letter * /*not used*/, T1 &arg1, T2 &arg2)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1, arg2));
            hdl_ = (Envelope)this;
        }

        template<typename T1>
        CWrapHandle(Letter * /*not used*/, T1 &arg1)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(arg1));
            hdl_ = (Envelope)this;
        }
#else
        //construct brand new by actually allocating into the pool
        //(the not-used argument only present for overload resolution
        //purposes, to differentiate from default constructor
        //since that one does something else)
        //
        template<typename...Args>
        explicit CWrapHandle(Letter & /*not used*/, Args...args)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(args...));
            hdl_ = (Envelope)this;
        }

        //same as above, but using pointer for convenience
        //DANGER here: Envelope is also a pointer type
        //so there will be an ambiguity (for some reason
        //ignored by compiler) between this constructor
        //and next (Envelope one) when using nullptr
        //
        //(pass (Letter*)NULL at invocation, NOT nullptr!!!)
        //
        template<typename...Args>
        explicit CWrapHandle(Letter * /*not used*/, Args...args)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(new Letter(args...));
            hdl_ = (Envelope)this;
        }
#endif

        CWrapHandle(Letter *ptr)
        {
            //Letters should be allocated directly on the heap:
            handled_ptr_.reset(ptr);
            hdl_ = (Envelope)this;
        }

        //construct out of previously allocated into the pool
        //
        explicit CWrapHandle(Envelope env) :
            hdl_(env)
        {
            CWrapHandle *convert = (CWrapHandle *)env;

            if ( !convert )
            {
                FatalError("Invalid/null C wrapper.\n", AMGX_ERR_BAD_PARAMETERS);    //throws...
            }

            //get_mem_manager<CWrapHandle>().allocate<CWrapHandle>(Letter)
            //     allocates both the wrapper and the Letter (via Letter constructor)
            //     each in its own pool;
            //     hence, the Letter gets allocated in get_mem_manager<Letter>()
            //     so, that's where this retrieval constructor must retrieve
            //     the Letter from, not from the CWrapHandle pool
            //handled_ptr_ = get_mem_manager<Letter>().template get_pointer<Letter, 0>((Letter*)env);
            //this will throw if not found in the pool:
            //
            //handled_ptr_ = get_mem_manager<Letter>().template get_pointer<Letter, 0>(convert->handled_ptr_.get());
            //Nope: Letters should be allocated directly on the heap:
            handled_ptr_ = convert->handled_ptr_;
            last_solve_status_ = convert->last_solve_status_;//PROBLEM: it's copy, it should be reference!
        }

        ~CWrapHandle(void)
        {
            handled_ptr_.reset();//it should be done by its own destructor
        }

        AMGX_STATUS last_solve_status(void) const
        {
            return last_solve_status_;
        }

        void set_last_solve_status(AMGX_STATUS st)
        {
            last_solve_status_ = st;
        }

        AMGX_STATUS &last_solve_status()
        {
            CWrapHandle *convert = (CWrapHandle *)hdl_;

            if ( !convert )
            {
                FatalError("Invalid/null C wrapper.\n", AMGX_ERR_BAD_PARAMETERS);    //throws...
            }

            return convert->last_solve_status_;
        }

        Envelope hdl(void) const
        {
            return hdl_;
        }

        //return by ref to eliminate needless
        //use_count increases:
        //
        const std::shared_ptr<Letter> &wrapped(void) const
        {
            return handled_ptr_;
        }

        //return by ref to eliminate needless
        //use_count increases:
        //
        std::shared_ptr<Letter> &wrapped(void)
        {
            return handled_ptr_;
        }

    private:
        Envelope hdl_;
        std::shared_ptr<Letter> handled_ptr_;
        AMGX_STATUS last_solve_status_;
};

namespace  //unnamed
{
#if 0
template<AMGX_Mode CASE,
         template<typename> class Letter,
         typename Envelope>
inline auto get_mode_object_from(Envelope envl) -> Letter<typename TemplateMode<CASE>::Type> *
{
    typedef Letter<typename TemplateMode<CASE>::Type> LetterT;
    typedef CWrapHandle<Envelope, LetterT> LetterW;
    LetterW letter(envl);
    return letter.wrapped().get();
}
#endif

#if 0
template<typename Envelope>
inline AMGX_Mode get_mode_from(const Envelope &envl)
{
    typename std::map < Envelope, int >::const_iterator itFound = get_mode_bookkeeper<Envelope>().find(envl);

    if (itFound == get_mode_bookkeeper<Envelope>().end())
    {
        //throws...
        //
        FatalError("Mode not found.\n", AMGX_ERR_BAD_MODE);
    }

    AMGX_Mode mode = static_cast<AMGX_Mode>(itFound->second);
    return mode;
}
#endif

#if defined ( _MSC_VER ) && _MSC_VER <= 1700
//MSVC c++11 workaround 

template<typename Letter,
         typename Envelope>
inline auto create_managed_object(Envelope *envl)
-> CWrapHandle<Envelope, Letter> *
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().allocate((Letter *)NULL).get();
    *envl = (Envelope)wrapper;
    return wrapper;
}

template<typename Letter,
         typename Envelope,
         typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5>
inline auto create_managed_object(Envelope *envl,
                                  const T1 &t1,
                                  const T2 &t2,
                                  const T3 &t3,
                                  const T4 &t4,
                                  const T5 &t5)
-> CWrapHandle<Envelope, Letter> *
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().allocate((Letter *)NULL, t1, t2, t3, t4, t5).get();
    *envl = (Envelope)wrapper;
    return wrapper;
}

template<typename Letter,
         typename Envelope,
         typename T1,
         typename T2,
         typename T3,
         typename T4>
inline auto create_managed_object(Envelope *envl,
                                  const T1 &t1,
                                  const T2 &t2,
                                  const T3 &t3,
                                  const T4 &t4)
-> CWrapHandle<Envelope, Letter> *
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().allocate((Letter *)NULL, t1, t2, t3, t4).get();
    *envl = (Envelope)wrapper;
    return wrapper;
}

template<typename Letter,
         typename Envelope,
         typename T1,
         typename T2,
         typename T3>
inline auto create_managed_object(Envelope *envl,
                                  const T1 &t1,
                                  const T2 &t2,
                                  const T3 &t3)
-> CWrapHandle<Envelope, Letter> *
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().allocate((Letter *)NULL, t1, t2, t3).get();
    *envl = (Envelope)wrapper;
    return wrapper;
}

template<typename Letter,
         typename Envelope,
         typename T1,
         typename T2>
inline auto create_managed_object(Envelope *envl,
                                  const T1 &t1,
                                  const T2 &t2)
-> CWrapHandle<Envelope, Letter> *
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().allocate((Letter *)NULL, t1, t2).get();
    *envl = (Envelope)wrapper;
    return wrapper;
}

template<typename Letter,
         typename Envelope,
         typename T1>
inline auto create_managed_object(Envelope *envl,
                                  const T1 &t1)
-> CWrapHandle<Envelope, Letter> *
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().allocate((Letter *)NULL, t1).get();
    *envl = (Envelope)wrapper;
    return wrapper;
}

//deallocator
//
template<typename Envelope,
         typename Letter>
inline bool remove_managed_object(Envelope envl)
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    //Either:find the wrapper based on envelope pointer
    //      and then erase the wrapper from the map...
    //      or
    //      try pointer conversion below
    //      (inverse of what create_managed_mode_object() does)
    //
    LetterW *letter = (LetterW *)(envl);
    bool flag = get_mem_manager<LetterW>().free(letter);
    //eliminate the ownership of the letter
    //inside the envelope:
    //
    ///letter->wrapped().reset();//nope...letter has been destructed
    return flag;
}

//deallocator for mode objects (other than Matrix, which requires more)
//
template<AMGX_Mode CASE,
         template<typename> class Letter,
         typename Envelope>
inline bool remove_managed_object(Envelope envl)
{
    typedef Letter<typename TemplateMode<CASE>::Type> LetterT;
    typedef CWrapHandle<Envelope, LetterT> LetterW;
    //Either:find the wrapper based on envelope pointer
    //      and then erase the wrapper from the map...
    //      or
    //      try pointer conversion below
    //      (inverse of what create_managed_mode_object() does)
    //
    LetterW *letter = (LetterW *)(envl);
    //mode objects must also be removed from the mode bookkeeping map:
    //
    size_t n_erased = get_mode_bookkeeper<Envelope>().erase(envl);
    bool flag = get_mem_manager<LetterW>().free(letter);
    //eliminate the ownership of the letter
    //inside the envelope:
    //
    ///letter->wrapped().reset();//nope...letter has been destructed
    return flag;
}

//deallocator for mode objects like Matrix, which requires more...
//
template<AMGX_Mode CASE>
inline bool remove_managed_matrix(AMGX_matrix_handle envl)
{
    typedef AMGX_matrix_handle Envelope;
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixT;
    typedef CWrapHandle<Envelope, MatrixT> MatrixW;
    //Either:find the wrapper based on envelope pointer
    //      and then erase the wrapper from the map...
    //      or
    //      try pointer conversion below
    //      (inverse of what create_managed_mode_object() does)
    //
    MatrixW *letter = (MatrixW *)(envl);
    //in addition:
    MatrixT *A = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(envl);

    if ( (A->is_manager_external()) && (A->manager != nullptr) )
    {
        delete A->manager;
    }

    //mode objects must also be removed from the mode bookkeeping map:
    //
    size_t n_erased = get_mode_bookkeeper<Envelope>().erase(envl);
    bool flag = get_mem_manager<MatrixW>().free(letter);
    //eliminate the ownership of the letter
    //inside the envelope:
    //
    ///letter->wrapped().reset();//nope...letter has been destructed
    return flag;
}
#else
// other compilers

#if 0
//factory method for objects depending on AMGX_Mode
//
template<AMGX_Mode CASE,
         template<typename> class Letter,
         typename Envelope,
         typename...CnstrArgs>
inline auto create_managed_mode_object(Envelope *envl,
                                       AMGX_Mode mode,
                                       CnstrArgs...cnstr_args)  -> CWrapHandle<Envelope, Letter<typename TemplateMode<CASE>::Type>> *
{
    typedef Letter<typename TemplateMode<CASE>::Type> LetterT;
    typedef CWrapHandle<Envelope, LetterT> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().template allocate<LetterW>(new LetterT(cnstr_args...)).get();
    wrapper->set_mode(mode);
    *envl = (Envelope)wrapper;
    return wrapper;
}
#endif

template<typename Letter,
         typename Envelope,
         typename...CnstrArgs>
inline auto create_managed_object(Envelope *envl,
                                  CnstrArgs...cnstr_args)  -> CWrapHandle<Envelope, Letter> *
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    LetterW *wrapper =
        get_mem_manager<LetterW>().template allocate<LetterW>(new Letter(cnstr_args...)).get();
    assert(wrapper != NULL);
    *envl = (Envelope)wrapper;
    return wrapper;
}

//deallocator
//
template<typename Envelope,
         typename Letter>
inline bool remove_managed_object(Envelope envl)
{
    typedef CWrapHandle<Envelope, Letter> LetterW;
    //Either:find the wrapper based on envelope pointer
    //      and then erase the wrapper from the map...
    //      or
    //      try pointer conversion below
    //      (inverse of what create_managed_mode_object() does)
    //
    LetterW *letter = (LetterW *)(envl);
    bool flag = get_mem_manager<LetterW>().template free<LetterW>(letter);
    //eliminate the ownership of the letter
    //inside the envelope:
    //
    ///letter->wrapped().reset();//nope...letter has been destructed
    return flag;
}

#if 0
//deallocator for mode objects (other than Matrix, which requires more)
//
template<AMGX_Mode CASE,
         template<typename> class Letter,
         typename Envelope>
inline bool remove_managed_object(Envelope envl)
{
    typedef Letter<typename TemplateMode<CASE>::Type> LetterT;
    typedef CWrapHandle<Envelope, LetterT> LetterW;
    //Either:find the wrapper based on envelope pointer
    //      and then erase the wrapper from the map...
    //      or
    //      try pointer conversion below
    //      (inverse of what create_managed_mode_object() does)
    //
    LetterW *letter = (LetterW *)(envl);
    //mode objects must also be removed from the mode bookkeeping map:
    //
    size_t n_erased = get_mode_bookkeeper<Envelope>().erase(envl);
    bool flag = get_mem_manager<LetterW>().template free<LetterW>(letter);
    //eliminate the ownership of the letter
    //inside the envelope:
    //
    ///letter->wrapped().reset();//nope...letter has been destructed
    return flag;
}
#endif
#endif

#if defined ( _MSC_VER ) && _MSC_VER <= 1700
//MSVC c++11 workaround 

//deallocator for mode objects like Matrix, which requires more...
//
template<AMGX_Mode CASE>
inline bool remove_managed_matrix(AMGX_matrix_handle envl)
{
    typedef AMGX_matrix_handle Envelope;
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixT;
    typedef CWrapHandle<Envelope, MatrixT> MatrixW;
    //Either:find the wrapper based on envelope pointer
    //      and then erase the wrapper from the map...
    //      or
    //      try pointer conversion below
    //      (inverse of what create_managed_mode_object() does)
    //
    MatrixW *letter = (MatrixW *)(envl);
    //in addition:
    MatrixT *A = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(envl);

    if ( (A->is_manager_external()) && (A->manager != nullptr) )
    {
        delete A->manager;
    }

    //mode objects must also be removed from the mode bookkeeping map:
    //
    size_t n_erased = get_mode_bookkeeper<Envelope>().erase(envl);
    bool flag = get_mem_manager<MatrixW>().free(letter);
    //eliminate the ownership of the letter
    //inside the envelope:
    //
    ///letter->wrapped().reset();//nope...letter has been destructed
    return flag;
}
#else

#if 0
//deallocator for mode objects like Matrix, which requires more...
//
template<AMGX_Mode CASE>
inline bool remove_managed_matrix(AMGX_matrix_handle envl)
{
    typedef AMGX_matrix_handle Envelope;
    typedef Matrix<typename TemplateMode<CASE>::Type> MatrixT;
    typedef CWrapHandle<Envelope, MatrixT> MatrixW;
    //Either:find the wrapper based on envelope pointer
    //      and then erase the wrapper from the map...
    //      or
    //      try pointer conversion below
    //      (inverse of what create_managed_mode_object() does)
    //
    MatrixW *letter = (MatrixW *)(envl);
    //in addition:
    MatrixT *A = get_mode_object_from<CASE, Matrix, AMGX_matrix_handle>(envl);

    if ( (A->is_manager_external()) && (A->manager != nullptr) )
    {
        delete A->manager;
    }

    //mode objects must also be removed from the mode bookkeeping map:
    //
    size_t n_erased = get_mode_bookkeeper<Envelope>().erase(envl);
    bool flag = get_mem_manager<MatrixW>().template free<MatrixW>(letter);
    //eliminate the ownership of the letter
    //inside the envelope:
    //
    ///letter->wrapped().reset();//nope...letter has been destructed
    return flag;
}
#endif

#endif

} //namespace unnamed


}//namespace amgx
