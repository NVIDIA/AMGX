/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

#if defined(_WIN32)
        #define _CRT_SECURE_NO_WARNINGS
        #define copystring _strdup
        #include <windows.h>
#else
        #define copystring strdup
        #include <unistd.h>
#endif

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <cuda.h>

#if !defined(__APPLE__)
#include <malloc.h>
#endif

#include <time.h>
#include <cusp/detail/timer.h>

#if defined(__ICC) || defined(__ICL)
        #pragma warning( disable: 1684 ) // (size_t )name >> 5
        #pragma warning( disable: 1011 ) // missing return statement at end of non-void function
#endif

#undef threadlocal

#if defined(_MSC_VER)
        #define YIELD() Sleep(0);
        #define PRINTFU64() "%I64u"
        #define PATHSLASH() '\\'
        #define threadlocal __declspec(thread)
        #define snprintf _snprintf

        #undef inline
        #define inline __forceinline
#else
        #include <sched.h>
        #define yield() sched_yield();
        #define printfu64() "%lu"
        #define PRINTFU64() "%lu"
        #define pathslash() '/'
        #define PATHSLASH() '/'
        #define threadlocal __thread
#endif

#if !defined(__PROFILER_SMP__)
        #undef threadlocal
        #define threadlocal
#endif

namespace cusp
{
namespace detail
{
namespace profiler
{

        size_t nextpow2( size_t x ) {
                x |= ( x >>  1 );
                x |= ( x >>  2 );
                x |= ( x >>  4 );
                x |= ( x >>  8 );
                x |= ( x >> 16 );
                return ( x + 1 );
        }

        template< class T >
        inline void zeroarray( T* array, size_t count ) {
                memset( array, 0, count * sizeof( T ) );
        }

        template< class T >
        inline T* makepointer( T* base, size_t byteoffset ) {
                return (T*)((const char *)base + byteoffset);
        }

        template< class T >
        inline void swapitems( T& a, T& b ) {
                T tmp = a;
                a = b;
                b = tmp;
        }

        #undef min
        #undef max

	struct CASLock {
		void Acquire() {}
		void Release() {}
		bool TryAcquire() { return false; }
		bool TryRelease() { return false; }
		size_t Value() const { return 0; }
		size_t dummy;
	};


        template< class T >
        inline const T& min( const T& a, const T& b ) {
                return ( a < b ) ? a : b;
        }

        template< class T >
        inline const T& max( const T& a, const T& b ) {
                return ( a < b ) ? b : a;
        }

        /*
        =============
        Buffer - Don't use for anything with a constructor/destructor. Doesn't shrink on popping
        =============
        */

        template< class T >
        struct Buffer 
	{
        protected:
                T* mBuffer;
                size_t mAlloc, mItems;

	public:
                Buffer() : mBuffer(NULL), mAlloc(0), mItems(0) { Resize( 4 ); }
                Buffer( size_t size ) : mBuffer(NULL), mAlloc(0), mItems(0) { Resize( size ); }

                ~Buffer() { free( mBuffer ); }

                void Clear() { mItems = ( 0 ); }
                T* Data() { return ( mBuffer ); }
                void EnsureCapacity( size_t capacity ) { if ( capacity >= mAlloc ) Resize( capacity * 2 ); }
                T* Last() { return ( &mBuffer[ mItems - 1 ] ); }
                void Push( const T& item ) { EnsureCapacity( mItems + 1 ); mBuffer[ mItems++ ] = ( item ); }
                T& Pop() { return ( mBuffer[ --mItems ] ); }

                void Resize( size_t newsize ) 
		{
                        mAlloc = nextpow2( newsize );
                        mBuffer = (T*)realloc( mBuffer, mAlloc * sizeof( T ) );
                }

                size_t Size() const { return mItems; }

                template< class Compare >
                void Sort( Compare comp ) 
		{
                        if ( mItems <= 1 )
                                return;
                       
                        Buffer scratch( mItems );

                        // merge sort with scratch buffer
                        T* src = Data();
			T* dst = scratch.Data();
                        for( size_t log = 2; log < mItems * 2; log *= 2 ) 
			{
                                T* out = dst;
                                for( size_t i = 0; i < mItems; i += log ) 
				{
                                        size_t lo = i, lo2 = min( i + log / 2, mItems );
                                        size_t hi = lo2, hi2 = min( lo + log, mItems );
                                        while ( ( lo < lo2 ) && ( hi < hi2 ) )
                                                *out++ = ( comp( src[lo], src[hi] ) ) ? src[lo++] : src[hi++];
                                        while ( lo < lo2 ) *out++ = src[lo++];
                                        while ( hi < hi2 ) *out++ = src[hi++];
                                }

                                swapitems( src, dst );
                        }

                        if ( src != mBuffer )
                                swapitems( mBuffer, scratch.mBuffer );
                }

                template< class Mapto >
                void ForEachByRef( Mapto &mapto, size_t limit ) 
		{
                        limit = ( limit < mItems ) ? limit : mItems;
                        size_t last = limit - 1;
                        for ( size_t i = 0; i < limit; ++i )
                                mapto( mBuffer[ i ], i == last );
                }

                template< class Mapto > void ForEach( Mapto mapto, size_t limit ) { ForEachByRef( mapto, limit ); }
                template< class Mapto > void ForEach( Mapto mapto ) { ForEachByRef( mapto, mItems ); }

                T& operator[] ( size_t index ) { return ( mBuffer[ index ] ); }
                const T& operator[] ( size_t index ) const { return ( mBuffer[ index ] ); }

        };      

        /*
        =============
        Caller
        =============
        */

        struct Caller 
	{

	protected:
                const char *mName;
                cusp::detail::timer mTimer;
                size_t mBucketCount, mNumChildren;
                Caller **mBuckets, *mParent;

                bool mActive;
                unsigned long mChildTicks;

        public:
                // caller
                static Buffer<char> mFormatter;

                // global
                static double mTimerOverhead, mRdtscOverhead;
                static double mGlobalDuration;

                static struct Max 
		{
		public:
                        enum f64Enum { SelfMs = 0, Ms, Avg, SelfAvg, f64Enums };
                        enum u64Enum { Calls = 0, TotalCalls, u64Enums };

                        void reset() 
			{
                                memset( this, 0, sizeof( *this ) );
                        }

                        void check( u64Enum e, unsigned long u ) { if ( u64fields[e] < u ) u64fields[e] = u; if ( e == Calls ) u64fields[TotalCalls] += u; }
                        void check( f64Enum e, double f ) { if ( f64fields[e] < f ) f64fields[e] = f; }

                        const unsigned long &operator() ( u64Enum e ) const { return u64fields[e]; }
                        const double &operator() ( f64Enum e ) const { return f64fields[e]; }

                protected:
                        unsigned long u64fields[u64Enums];
                        double f64fields[f64Enums];

                } maxStats;

                // per thread state
                struct ThreadState 
		{
                        CASLock threadLock;
                        bool requireThreadLock;
                        Caller *activeCaller;
                };
               
                static threadlocal ThreadState thisThread;

                struct foreach 
		{
                        // Adds each Caller to the specified buckets
                        struct AddToNewBuckets 
			{
                                Caller **mBuckets;
                                size_t mBucketCount;

                                AddToNewBuckets( Caller **buckets, size_t bucket_count ) : mBuckets(buckets), mBucketCount(bucket_count) {}

                                void operator()( Caller *item ) 
				{
                                        FindEmptyChildSlot( mBuckets, mBucketCount, item->mName ) = item;
                                }
                        };


                        // Destructs a Caller
                        struct Deleter 
			{
                                void operator()( Caller *item ) 
				{
                                        delete item;
                                }
                        };

                        // Merges a Caller with the root
                        struct Merger 
			{
                                Caller *mRoot;

                                Merger( Caller *root ) : mRoot(root) {}

                                void addFrom( Caller *item ) 
				{ 
					(*this)( item ); 
				}

                                void operator()( Caller *item ) 
				{
                                        Caller *child = mRoot->FindOrCreate( item->GetName() );
                                        child->GetTimer() += item->GetTimer();
                                        child->SetParent( item->GetParent() );
                                        item->ForEachNonEmpty( Merger( child ) );
                                }
                        };

                        // Prints a Caller
                        struct Printer 
			{
                                size_t mIndent;

                                Printer( size_t indent ) : mIndent(indent) {}

                                void operator()( Caller *item, bool islast ) const 
				{
                                        item->Print( mIndent, islast );
                                }
                        };

                        // Sums Caller's milliseconds
                        struct SumMilliseconds 
			{
                                double sum;

                                SumMilliseconds() : sum(0) {}

                                void operator()( Caller *item ) 
				{
                                        sum += ( item->mTimer.milliseconds );
                                }
                        };

			
                        struct SoftReset 
			{ 
                                void operator()( Caller *item ) 
				{ 
                                        item->GetTimer().soft_reset();
                                        //item->ForEach( soft_reset() );
                                } 
                        };


                        struct UpdateTopMaxStats 
			{
                                UpdateTopMaxStats() { maxStats.reset(); }

                                void operator()( Caller *item, bool islast ) 
				{
                                        if ( !item->GetParent() )
                                                return;
                                        maxStats.check( Max::Calls, item->mTimer.calls );
                                }
                        };

                }; // foreach


                struct compare 
		{
                        struct Milliseconds 
			{
                                bool operator()( const Caller *a, const Caller *b ) const 
				{
                                        return ( a->mTimer.milliseconds > b->mTimer.milliseconds );
                                }
                        };

                        struct SelfTicks 
			{
                                bool operator()( const Caller *a, const Caller *b ) const 
				{
                                        return ( ( a->mTimer.milliseconds - a->mChildTicks ) > ( b->mTimer.milliseconds - b->mChildTicks ) );
                                }
                        };

                        struct Calls 
			{
                                bool operator()( const Caller *a, const Caller *b ) const 
				{
                                        return ( a->mTimer.calls > b->mTimer.calls );
                                }
                        };
                }; // sort


                /*
                 *       Since Caller.mTimer.ticks is inclusive of all children, summing the first level
                 *       children of a Caller to Caller.mChildTicks is an accurate total of the complete
                 *       child tree.
		 *
                 *       mTotals is used to keep track of total ticks by Caller excluding children
                 */
                struct ComputeChildTicks 
		{
                        Caller &mTotals;

                        ComputeChildTicks( Caller &totals ) : mTotals(totals) { maxStats.reset(); }

                        void operator()( Caller *item ) 
			{
                                foreach::SumMilliseconds sumchildren;
                                item->ForEachByRefNonEmpty( sumchildren );
                                item->mChildTicks = ( sumchildren.sum );

                                double selfticks = ( item->mTimer.milliseconds >= item->mChildTicks ) ? ( item->mTimer.milliseconds - item->mChildTicks ) : 0.0;
                                Caller &totalitem = ( *mTotals.FindOrCreate( item->mName ) );
                                totalitem.mTimer.milliseconds += selfticks;
                                totalitem.mTimer.calls += item->mTimer.calls;
                                totalitem.SetParent( item->GetParent() );

                                // don't include the root node in the max stats
                                if ( item->GetParent() ) 
				{
                                        maxStats.check( Max::SelfMs, selfticks );
                                        maxStats.check( Max::Calls, item->mTimer.calls );
                                        maxStats.check( Max::Ms, item->mTimer.milliseconds );
                                }

                                // compute child ticks for all children of children of this caller
                                item->ForEachByRefNonEmpty( *this );
                        }
                };

                /*
                 *  Format a Caller's information. ComputeChildTicks will need to be used on the Root
                 *  to generate mChildTicks for all Callers
                 */
                struct Format 
		{
                        const char *mPrefix;

                        Format( const char *prefix ) : mPrefix(prefix) {}

                        void operator()( Caller *item, bool islast ) const 
			{
                                double ms = item->mTimer.milliseconds;
				const char * hyphen = strrchr(item->mName,'(');
				int size = hyphen-item->mName;
				
                                printf( "%s %.2f ms, %lu calls: %.*s\n",
                                        mPrefix, ms, item->mTimer.calls, size, item->mName );
                        }
                };

                /*
                        Methods
                */

                // we're guaranteed to be null because of calloc. ONLY create Callers with "new"!
                Caller( const char *name, Caller *parent = NULL ) 
		{
                        mName = name;
                        mParent = parent;
                        Resize( 2 ); // mBuckets must always exist and mBucketCount >= 2!
                }
               
                ~Caller() 
		{
                        ForEach( foreach::Deleter() );
                        free( mBuckets );
                }

                void CopyToListNonEmpty( Buffer<Caller *> &list ) 
		{
                        list.Clear();

                        for ( size_t i = 0; i < mBucketCount; ++i )
                                if ( mBuckets[ i ] && !mBuckets[ i ]->GetTimer().is_empty() )
                                        list.Push( mBuckets[ i ] );
                }

                inline Caller *FindOrCreate( const char *name ) 
		{
                        size_t index = ( GetBucket( name, mBucketCount ) ), mask = ( mBucketCount - 1 );
                        for ( Caller *caller = mBuckets[index]; caller; caller = mBuckets[index & mask] ) 
			{
                                if ( caller->mName == name )
                                        return caller;
                               
                                index = ( index + 1 );
                        }

                        // didn't find the caller, lock this thread and mutate
                        EnsureCapacity( ++mNumChildren );
                        Caller *&slot = FindEmptyChildSlot( mBuckets, mBucketCount, name );
                        slot = new Caller( name, this );
                        return slot;
                }

                template< class Mapto >
                void ForEachByRef( Mapto &mapto ) 
		{
                        for ( size_t i = 0; i < mBucketCount; ++i )
                                if ( mBuckets[ i ] )
                                        mapto( mBuckets[ i ] );
                }

                template< class Mapto >
                void ForEachByRefNonEmpty( Mapto &mapto ) 
		{
                        for ( size_t i = 0; i < mBucketCount; ++i )
                                if ( mBuckets[ i ] && !mBuckets[ i ]->GetTimer().is_empty() )
                                        mapto( mBuckets[ i ] );
                }

                template< class Mapto >
                void ForEach( Mapto mapto ) 
		{
                        ForEachByRef( mapto );
                }

                template< class Mapto >
                void ForEachNonEmpty( Mapto mapto ) 
		{
                        ForEachByRefNonEmpty( mapto );
                }

                inline Caller *GetParent() 
		{
                        return mParent;
                }

                cusp::detail::timer &GetTimer() 
		{
                        return mTimer;
                }

                const char *GetName() const 
		{
                        return mName;
                }

                bool IsActive() const 
		{
                        return mActive;
                }

                void Print( size_t indent = 0, bool islast = false ) 
		{
                        Buffer<Caller *> children( mNumChildren );
                        CopyToListNonEmpty( children );

                        mFormatter.EnsureCapacity( indent + 3 );
                        char *fmt = ( &mFormatter[indent] );
                       
                        if ( indent ) 
			{
                                fmt[-2] = ( islast ) ? ' ' : '|';
                                fmt[-1] = ( islast ) ? '\\' : ' ';
                        }

                        fmt[0] = ( children.Size() ) ? '+' : '-';
                        fmt[1] = ( '-' );
                        fmt[2] = ( 0 );
                       
                        Format(mFormatter.Data())( this, islast );

                        if ( indent && islast )
                                fmt[-2] = fmt[-1] = ' ';

                        if ( children.Size() ) 
			{
                                children.Sort( compare::Milliseconds() );
                                children.ForEach( foreach::Printer(indent+2) );
                        }
                }

                void PrintTopStats( size_t nitems ) 
		{
                        nitems = ( nitems > mNumChildren ) ? mNumChildren : nitems;
                        printf( "\ntop %lu functions (self time)\n", (size_t )nitems );
                        Buffer<Caller *> sorted( mNumChildren );
                        CopyToListNonEmpty( sorted );
                        sorted.Sort( compare::SelfTicks() );
                        sorted.ForEach( Format(">"), nitems );
                }

                void Resize( size_t new_size ) 
		{
                        new_size = ( new_size < mBucketCount ) ? mBucketCount << 1 : nextpow2( new_size - 1 );
                        Caller **new_buckets = (Caller **)calloc( new_size, sizeof( Caller* ) );
                        ForEach( foreach::AddToNewBuckets( new_buckets, new_size ) );

                        free( mBuckets );
                        mBuckets = ( new_buckets );
                        mBucketCount = ( new_size );
                }

                void Reset() 
		{
                        ForEach( foreach::Deleter() );
                        zeroarray( mBuckets, mBucketCount );
                        mNumChildren = ( 0 );
                        mTimer.reset();                
                }

                void SetActive( bool active ) 
		{
                        mActive = active;
                }

                void SetParent( Caller *parent ) 
		{
                        mParent = parent;
                }

		void SoftReset() 
		{
                        mTimer.soft_reset();
                        ForEach( foreach::SoftReset() );
                }

                void Start() 
		{
                        mTimer.start();
                }

                void Stop() 
		{
                        mTimer.stop();
                }

                void *operator new ( size_t size ) 
		{
                        return calloc( size, 1 );
                }

                void operator delete ( void *p ) 
		{
                        free( p );
                }

        protected:
                static inline Caller *&FindEmptyChildSlot( Caller **buckets, size_t bucket_count, const char *name ) 
		{
                        size_t index = ( GetBucket( name, bucket_count ) ), mask = ( bucket_count - 1 );
                        Caller **caller = &buckets[index];

                        for ( ; *caller; caller = &buckets[index & mask] )
                                index = ( index + 1 );

                        return *caller;
                }

                inline static size_t GetBucket( const char *name, size_t bucket_count ) 
		{
                        return size_t( ( ( (size_t )name >> 5 ) /* * 2654435761 */ ) & ( bucket_count - 1 ) );
                }

                inline void EnsureCapacity( size_t capacity ) 
		{
                        if ( capacity < ( mBucketCount / 2 ) )
                                return;
                        Resize( capacity );
                }

        };


	#if defined(__PROFILER_ENABLED__)
        threadlocal Caller::ThreadState Caller::thisThread = { {0}, 0, 0 };
        double Caller::mTimerOverhead = 0.0;
        double Caller::mGlobalDuration = 0.0;
        Caller::Max Caller::maxStats;
        Buffer<char> Caller::mFormatter( 64 );
        char *programName = NULL, *commandLine = NULL;

        void detectByArgs( int argc, const char *argv[] ) 
	{
                const char *path = argv[0], *finalSlash = path, *iter = path;
                for ( ; *iter; ++iter )
                        finalSlash = ( *iter == PATHSLASH() ) ? iter + 1 : finalSlash;
                if ( !*finalSlash )
                        finalSlash = path;
                programName = copystring( finalSlash );
               
                size_t width = 0;
                for ( int i = 1; i < argc; i++ ) 
		{
                        size_t len = strlen( argv[i] );
                        commandLine = (char *)realloc( commandLine, width + len + 1 );
                        memcpy( commandLine + width, argv[i], len );
                        commandLine[width + len] = ' ';
                        width += len + 1;
                }
                if ( width )
                        commandLine[width - 1] = '\x0';
        }

        void detectWinMain( const char *cmdLine ) 
	{
	#if defined(_MSC_VER)
                char path[1024], *finalSlash = path, *iter = path;
                GetModuleFileName( NULL, path, 1023 );
                for ( ; *iter; ++iter )
                        finalSlash = ( *iter == PATHSLASH() ) ? iter + 1 : finalSlash;
                if ( !*finalSlash )
                        finalSlash = path;
                programName = copystring( finalSlash );
                commandLine = copystring( cmdLine );
	#else
                programName = copystring( "only_for_win32" );
                commandLine = copystring( "" );
	#endif
        }

        /*
        ============
        Root - Holds the root caller and the thread state for a thread
        ============
        */

        struct Root 
	{
                Caller *root;
                Caller::ThreadState *threadState;

                Root( Caller *caller, Caller::ThreadState *ts ) : root(caller), threadState(ts) {}
        };

        struct GlobalThreadList {
                ~GlobalThreadList() {
                        if ( list ) {
                                Buffer<Root> &threadsref = *list;
                                size_t cnt = threadsref.Size();
                                for ( size_t i = 0; i < cnt; i++ )
                                        delete threadsref[i].root;
                        }
                        delete list;
                }

                void AcquireGlobalLock() 
		{
                        threadsLock.Acquire();
                        if ( !list )
                                list = new Buffer<Root>;
                }

                void ReleaseGlobalLock() 
		{
                        threadsLock.Release();
                }

                Buffer<Root> *list;
                CASLock threadsLock;
        };

        cudaEvent_t globalStart;
        GlobalThreadList threads = { NULL, {0} };
        threadlocal Caller *root = NULL;
       

        /*
                Thread Dumping
        */

        struct PrintfDumper 
	{
                void Init() {}
                void Finish() {}

                void GlobalInfo( float rawDuration ) 
		{
                        printf( "> Raw run time %.2f milliseconds\n", rawDuration );
                }

                void ThreadsInfo( unsigned long totalCalls, double timerOverhead ) 
		{
                        printf( "> Total calls " PRINTFU64() ", per call overhead %.2f msecs, estimated overhead %.2f msecs\n\n",
                                totalCalls, timerOverhead, timerOverhead * totalCalls );
                }

                void PrintThread( Caller *root ) 
		{
                        root->Print();
                        printf( "\n\n" );
                }

                void PrintAccumulated( Caller *accumulated ) 
		{
                        accumulated->PrintTopStats( 50 );
                }

        };

        template< class Dumper >
        void dumpThreads( Dumper dumper ) {
                float rawDuration;
    		cudaEvent_t end;
		cudaEventCreate(&end);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&rawDuration, globalStart, end);

                Caller *accumulate = new Caller( "/Top Callers" ), *packer = new Caller( "/Thread Packer" );
                Buffer<Caller *> packedThreads;

                dumper.Init();
                dumper.GlobalInfo( rawDuration );

                threads.AcquireGlobalLock();    

                // crawl the list of theads and store their data in to packer
                Buffer<Root> &threadsref = *threads.list;
                for ( size_t i = 0; i < threadsref.Size(); i++ ) {
                        Root &thread = threadsref[i];

                        // if the thread is no longer active, the lock won't be valid
                        bool active = ( thread.root->IsActive() );
                        if ( active ) {
                                thread.threadState->threadLock.Acquire();
                                // disable requiring our local lock in case the caller is in our thread, accumulate will try to set it otherwise
                                Caller::thisThread.requireThreadLock = false;
                                for ( Caller *walk = thread.threadState->activeCaller; walk; walk = walk->GetParent() )
                                        walk->GetTimer().soft_stop();
                        }

                        // create a dummy entry for each thread (fake a name with the address of the thread root)
                        Caller *stub = packer->FindOrCreate( (const char *)thread.root );
                        Caller::foreach::Merger( stub ).addFrom( thread.root );
                        Caller *stubroot = stub->FindOrCreate( thread.root->GetName() );
                        stubroot->SetParent( NULL ); // for proper crawling
                        packedThreads.Push( stubroot );

                        if ( active ) {
                                Caller::thisThread.requireThreadLock = true;
                                thread.threadState->threadLock.Release();
                        }
                }

                // working on local data now, don't need the threads lock any more
                threads.ReleaseGlobalLock();    

                // do the pre-computations on the gathered threads
                Caller::ComputeChildTicks preprocessor( *accumulate );
                for ( size_t i = 0; i < packedThreads.Size(); i++ )
                        preprocessor( packedThreads[i] );

                dumper.ThreadsInfo( Caller::maxStats( Caller::Max::TotalCalls ), Caller::mTimerOverhead );

                // print the gathered threads
                double sumMilliseconds = 0.0;
                for ( size_t i = 0; i < packedThreads.Size(); i++ ) {
                        Caller *root = packedThreads[i];
			double threadMilliseconds = root->GetTimer().milliseconds;
                        sumMilliseconds += threadMilliseconds;
                        Caller::mGlobalDuration = threadMilliseconds;
                        dumper.PrintThread( root );
                }

                // print the totals, use the summed total of ticks to adjust percentages
                Caller::mGlobalDuration = sumMilliseconds;
                dumper.PrintAccumulated( accumulate );          
                dumper.Finish();

                delete accumulate;
                delete packer;
        }

        void resetThreads() 
	{
        	cudaEventDestroy(globalStart);
        	cudaEventCreate(&globalStart); 

                if ( root )
                        root->SoftReset();
        }

        void enterThread( const char *name ) 
	{
                Caller *tmp = new Caller( name );

                threads.AcquireGlobalLock();
                threads.list->Push( Root( tmp, &Caller::thisThread ) );

                Caller::thisThread.activeCaller = tmp;
                tmp->Start();
                tmp->SetActive( true );
                root = tmp;

                threads.ReleaseGlobalLock();
        }

        void exitThread() 
	{
                threads.AcquireGlobalLock();

                root->Stop();
                root->SetActive( false );
                Caller::thisThread.activeCaller = NULL;

                threads.ReleaseGlobalLock();
        }

        inline void fastcall enterCaller( const char *name ) 
	{
                Caller *parent = Caller::thisThread.activeCaller;
                if ( !parent )
                        return;
               
                Caller *active = parent->FindOrCreate( name );
                active->Start();
                Caller::thisThread.activeCaller = active;
        }

        inline void exitCaller() 
	{
                Caller *active = Caller::thisThread.activeCaller;
                if ( !active )
                        return;
               
                active->Stop();
                Caller::thisThread.activeCaller = active->GetParent();
        }

        inline void pauseCaller() 
	{
                Caller *iter = Caller::thisThread.activeCaller;
                for ( ; iter; iter = iter->GetParent() )
                        iter->GetTimer().pause();
        }

        inline void unpauseCaller() 
	{
                Caller *iter = Caller::thisThread.activeCaller;
                for ( ; iter; iter = iter->GetParent() )
                        iter->GetTimer().unpause();
        }

        // enter the main thread automatically
        struct MakeRoot 
	{
                MakeRoot() 
		{
                        // get an idea of how long timer calls / rdtsc takes
                        const size_t reps = 1000;
                        for ( size_t tries = 0; tries < 20; tries++ ) 
			{
                                cusp::detail::timer t1, t2;
                                t1.start();
                                for ( size_t i = 0; i < reps; i++ ) 
				{
                                        t2.start();
                                        t2.stop();
                                }
                                t1.stop();
                                double avg = double(t2.milliseconds)/double(reps);
                                avg = double(t1.milliseconds)/double(reps);
                                Caller::mTimerOverhead = avg;
                        }

        		cudaEventCreate(&globalStart); 
        		cudaEventRecord(globalStart,0);
                        enterThread( "/Main" );
                }

                ~MakeRoot() 
		{
                        free( programName );
                        free( commandLine );
                }
        } makeRoot;

        void detect( int argc, const char *argv[] ) { detectByArgs( argc, argv ); }
        void detect( const char *commandLine ) { detectWinMain( commandLine ); }
        void dump() { dumpThreads( PrintfDumper() ); }
        void fastcall enter( const char *name ) { enterCaller( name ); }
        void fastcall exit() { exitCaller(); }
        void fastcall pause() { pauseCaller(); }
        void fastcall unpause() { unpauseCaller(); }
        void reset() { resetThreads(); }
	#else
        void detect( int argc, const char *argv[] ) {}
        void detect( const char *commandLine ) {}
        void dump() {}
        void fastcall enter( const char *name ) {}
        void fastcall exit() {}
        void fastcall pause() {}
        void fastcall unpause() {}
        void reset() {}
	#endif

} // end namespace profiler
} // end namespace detail
} // end namespace cusp

