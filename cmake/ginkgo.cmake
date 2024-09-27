# SPDX-FileCopyrightText: 2024 OGL authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include(FetchContent)

if(NOT ${OGL_USE_EXTERNAL_GINKGO})
  FetchContent_Declare(
    Ginkgo
    SYSTEM
    QUITE
    GIT_SHALLOW ON
    GIT_REPOSITORY "https://github.com/ginkgo-project/ginkgo.git"
    GIT_TAG ${GINKGO_CHECKOUT_VERSION})

  FetchContent_MakeAvailable(Ginkgo)
endif()
