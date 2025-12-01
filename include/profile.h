// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

void profileLevelUp();
void profileLevelDown();
void profileLevelZero();
void profilePhaseSetup();
void profilePhaseSolve();
void profilePhaseNone();
void profileSubphaseMatrixColoring();
void profileSubphaseSmootherSetup();
void profileSubphaseFindAggregates();
void profileSubphaseComputeRestriction();
void profileSubphaseComputeCoarseA();
void profileSubphaseNone();
void profileSubphaseTruncateP();
