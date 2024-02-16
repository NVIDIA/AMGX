// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
namespace amgx
{
template <class T_Config> class AMG_Level;
}

#include <amg.h>
#include <solvers/solver.h>
#include <cycles/cycle.h>
#include <amg_config.h>
#include <amgx_timer.h>
#include <vector>
#include <cassert>
#include <thread_manager.h>

namespace amgx
{

// async work class for smoother
template<class TConfig>
class AsyncSolverSetupTask : public AsyncTask
{
        Solver<TConfig> *solver;
        Matrix<TConfig> *matrix;

    public:
        AsyncSolverSetupTask(Solver<TConfig> *s, Matrix<TConfig> *m) : solver(s), matrix(m) {}

        bool terminate() { return false; }

        void exec()
        {
            solver->setup(*matrix, false);
        }
};

template< typename TConfig, AMGX_MemorySpace SrcSpace, AMGX_MemorySpace DstSpace, class CycleDispatcher >
class AMG_GenerateNextCycles {};

/********************************************************
 * AMG Level class:
 *  This class is a base class for AMG levels.  This
 *  class is a linked list of levels where each
 *  level contains the solution state for that level.
 ********************************************************/
template <class T_Config>
class AMG_Level
{
    public:
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        DEFINE_VECTOR_TYPES

        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;
        static const AMGX_MemorySpace other_memspace = MemorySpaceMap<opposite_memspace<TConfig::memSpace>::memspace>::id;
        typedef TemplateConfig<other_memspace, vecPrec, matPrec, indPrec> TConfig1;
        typedef TConfig1 T_Config1;


        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::VecPrec ValueTypeB;

        typedef typename TConfig::IndPrec IndexType;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;

        friend class AMG<vecPrec, matPrec, indPrec>;
        friend class AMG_Level<TConfig1>;

        AMG_Level(AMG_Class *amg, ThreadManager *tmng = NULL);
        virtual ~AMG_Level();

        virtual void restrictResidual(VVector &r, VVector &rr) = 0;
        virtual void prolongateAndApplyCorrection(VVector &c, VVector &bc, VVector &x, VVector &tmp) = 0;
        virtual void createCoarseVertices() = 0;
        virtual void createCoarseMatrices() = 0;
        virtual bool isClassicalAMGLevel() = 0;
        virtual IndexType getNumCoarseVertices() = 0;

        virtual void prepareNextLevelMatrix(const Matrix<TConfig> &A, Matrix<TConfig> &Ac) = 0;
        virtual void consolidateVector(VVector &r) = 0;
        virtual void unconsolidateVector(VVector &r) = 0;

        virtual void transfer_level(AMG_Level<TConfig1> *ref_lvl) = 0;

        void transfer_from(AMG_Level<TConfig1> *ref_lvl); // copy from other memoryspace
        void setup();
        void setup_smoother();

        inline bool isFinest() { return level_id == 0; }
        inline bool isCoarsest() { return next_d == NULL && next_h == NULL; }
        inline bool isNextCoarsest( ) { return ( next_d && next_d->isCoarsest( ) ) || ( next_h && next_h->isCoarsest( ) ); }
        inline bool isSetup() { return is_setup; }

        void setInitCycle() {init = true;}
        void setNextInitCycle() { if ( next_h ) next_h->setInitCycle( ); if ( next_d ) next_d->setInitCycle( ); }
        void unsetInitCycle() {init = false;}
        void unsetNextInitCycle() { if ( next_h ) next_h->unsetInitCycle( ); if ( next_d ) next_d->unsetInitCycle( ); }
        bool isInitCycle() {return init;}

        inline IndexType getNextLevelSize(void) const
        {
            return m_next_level_size;
        }

        template<class CycleDispatcher >
        void generateNextCycles( AMG_Class *amg, VVector &b, VVector &x, const CycleDispatcher &dispatcher = CycleDispatcher( ) )
        {
            static const AMGX_MemorySpace memSpace = TConfig::memSpace;

            if ( next_h != 0 )
            {
                AMG_GenerateNextCycles<TConfig, memSpace, AMGX_host, CycleDispatcher>::generate( amg, next_h, b, x, dispatcher );
            }
            else
            {
                AMG_GenerateNextCycles<TConfig, memSpace, AMGX_device, CycleDispatcher>::generate( amg, next_d, b, x, dispatcher );
            }
        }

        void launchCoarseSolver( AMG_Class *amg, VVector &b, VVector &x);

        inline int getLevelIndex() { return level_id; }
        inline void setLevelIndex( int index ) { level_id = index; tag = index + 1;}

        inline Matrix<TConfig> &getA() { return *A; }
        inline void setA(Matrix<TConfig> &a)
        {
            if (A == Aoriginal) {delete A; Aoriginal = NULL;}

            A = &a;
        }
        inline VVector &getbc() { return bc; }
        inline VVector &getxc() { return xc; }
        inline VVector &getr() { return r; }
        inline Solver<TConfig> *getSmoother() { return smoother; }

        inline const AMG_Level<TConfig_h> *getNextLevel( host_memory ) const { return next_h; }
        inline AMG_Level<TConfig_h> *&getNextLevel( host_memory ) { return next_h; }
        inline void setNextLevel( AMG_Level<TConfig_h> *level ) { next_h = level; }
        inline void resetNextLevel( host_memory ) { next_h = 0L; }
        inline void deleteNextLevel( host_memory ) { delete next_h; next_h = 0L; }

        inline const AMG_Level<TConfig_d> *getNextLevel( device_memory ) const { return next_d; }
        inline AMG_Level<TConfig_d> *&getNextLevel( device_memory ) { return next_d; }
        inline void setNextLevel( AMG_Level<TConfig_d> *level ) { next_d = level; }
        inline void resetNextLevel( device_memory ) { next_d = 0L; }
        inline void deleteNextLevel( device_memory ) { delete next_d; next_d = 0L; }
        inline bool isConsolidationLevel() { return m_is_consolidation_level; }
        inline bool isRootPartition() const { return m_is_root_partition; }
        inline void setIsConsolidationLevel(bool is_consolidation_level) { m_is_consolidation_level = is_consolidation_level; }
        inline bool isReuseLevel() { return m_is_reuse_level; }
        inline void setReuseLevel(bool is_reuse_level) { m_is_reuse_level = is_reuse_level; }

        inline size_t getNumRows() { return A->get_num_rows(); }

        //todo add smoother allocation also...
        inline size_t bytes(bool device_only = false)
        {
            size_t size = 0;

            if (A != NULL) { size += A->bytes(device_only); }

            if (Aoriginal != NULL) { size += Aoriginal->bytes(device_only); }

            size += bc.bytes(device_only) + xc.bytes(device_only) + r.bytes(device_only);
            return size;
        }

        levelProfile Profile;
        int tag;
        int is_setup;

        // Sets the solver name
        inline void setName(std::string &amg_level_name) { m_amg_level_name = amg_level_name; }

        // Returns the name of the solver
        inline std::string getName() const { return m_amg_level_name; }

    protected:
        std::vector<int> originalRow;
        std::vector<int> getOriginalRows();

    protected:
        Solver<TConfig> *smoother;
        Matrix<TConfig> *A;
        Matrix<TConfig> *Aoriginal;
        VVector bc, xc, r;

        AMG_Class *amg;
        AMG_Level<TConfig_h> *next_h;
        AMG_Level<TConfig_d> *next_d;
        int level_id;
        IndexType m_next_level_size;
        bool init;   //marks if the x vector needs to be initialized
        bool m_is_consolidation_level;
        bool m_is_reuse_level;
        std::string m_amg_level_name;


        bool m_is_root_partition;
        IndexType m_destination_part;
        INDEX_TYPE m_num_parts_to_consolidate;

};

template< typename TConfig, AMGX_MemorySpace MemSpace, class CycleDispatcher >
class AMG_GenerateNextCycles<TConfig, MemSpace, MemSpace, CycleDispatcher>
{
    public:
//  typedef typename TraitsFromMatrix<TConfig>::Traits MatrixTraits;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef CycleDispatcher CycleDispatcher_Class;
        typedef Vector<typename TConfig::template setMemSpace<MemSpace>::Type> Vector;

        static void generate( AMG_Class *amg, AMG_Level<TConfig> *level, Vector &b, Vector &x, const CycleDispatcher_Class &dispatcher )
        {
            typedef typename TConfig::MemSpace MemorySpace;
            Solver<TConfig> *coarseSolver = amg->getCoarseSolver( MemorySpace() );

            if ( level->isCoarsest( ) && coarseSolver )
            {
                level->launchCoarseSolver( amg, b, x );
            }
            else
            {
                dispatcher.dispatch( amg, level, b, x );
            }
        }
};

template< typename TConfig, class CycleDispatcher >
class AMG_GenerateNextCycles<TConfig, AMGX_host, AMGX_device, CycleDispatcher>
{
    public:
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef CycleDispatcher CycleDispatcher_Class;
        typedef typename MemorySpaceMap<AMGX_device>::Type device_memory;
        typedef TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, vecPrec, matPrec, indPrec> TConfig_d;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;

        static void generate( AMG_Class *amg, AMG_Level<TConfig_d> *level, Vector_h &b, Vector_h &x, const CycleDispatcher_Class &dispatcher )
        {
            Solver<TConfig_d> *coarseSolver = amg->getCoarseSolver( device_memory( ) );
            Vector_d b_d(b), x_d(x);

            if ( level->isCoarsest( ) && coarseSolver )
            {
                level->launchCoarseSolver( amg, b_d, x_d );
            }
            else
            {
                dispatcher.dispatch( amg, level, b_d, x_d );
            }

            x.copy(x_d);
        }
};

template< typename TConfig, class CycleDispatcher >
class AMG_GenerateNextCycles<TConfig, AMGX_device, AMGX_host, CycleDispatcher>
{
    public:
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef typename MemorySpaceMap<AMGX_host>::Type host_memory;
        typedef TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, vecPrec, matPrec, indPrec> TConfig_d;

        typedef Matrix<TConfig_h> Matrix_h;
        typedef Matrix<TConfig_d> Matrix_d;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;

        static void generate( AMG_Class *amg, AMG_Level<TConfig_h> *level, Vector_d &b, Vector_d &x, const CycleDispatcher &dispatcher )
        {
            Solver<TConfig_h> *coarseSolver = amg->getCoarseSolver( host_memory( ) );
            Vector_h b_h(b), x_h(x);

            if ( level->isCoarsest( ) && coarseSolver )
            {
                level->launchCoarseSolver( amg, b_h, x_h );
            }
            else
            {
                dispatcher.dispatch( amg, level, b_h, x_h );
            }

            x.copy(x_h);
        }
};

template<class TConfig>
class AMG_LevelFactory
{
    public:
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;

        virtual AMG_Level<TConfig> *create(AMG_Class *amg, ThreadManager *tmng = NULL) = 0;
        virtual ~AMG_LevelFactory() {};

        /********************************************
         * Register a convergence class with key "name"
         *******************************************/
        static void registerFactory(AlgorithmType name, AMG_LevelFactory<TConfig> *f);

        /********************************************
         * Unregister a convergence class with key "name"
         *******************************************/
        static void unregisterFactory(AlgorithmType name);

        /********************************************
         * Unregister all the solver classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates solvers based on cfg
        *********************************************/
        static AMG_Level<TConfig> *allocate(AMG_Class *amg, ThreadManager *tmng = NULL);

        typedef typename std::map<AlgorithmType, AMG_LevelFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<AlgorithmType, AMG_LevelFactory<TConfig>*> &getFactories( );
};

} // namespace amg
