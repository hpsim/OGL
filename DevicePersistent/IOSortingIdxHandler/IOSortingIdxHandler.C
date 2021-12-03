
#include <tuple>
#include "IOSortingIdxHandler.H"


namespace Foam {

IOSortingIdxHandler::IOSortingIdxHandler(const objectRegistry &db,
                                         const label nElems)
    : nElems_(nElems), is_sorted_(false){};


void IOSortingIdxHandler::init_sorting_idxs(){};


}  // namespace Foam
