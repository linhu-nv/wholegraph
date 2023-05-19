/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <c10/core/ScalarType.h>

#include "data_type.h"

namespace whole_graph {

namespace pytorch {

whole_graph::WMType C10ScalarToWMType(c10::ScalarType st);
c10::ScalarType WMTypeToC10Scalar(whole_graph::WMType wmt);

}// namespace pytorch

}// namespace whole_graph