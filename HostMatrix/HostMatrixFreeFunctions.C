#include "HostMatrix.H"

namespace Foam {

const lduInterfaceField *interface_getter(
    const lduInterfaceFieldPtrsList &interfaces, const label i)
{
#ifdef WITH_ESI_VERSION
    return interfaces.get(i);
#else
    return interfaces.operator()(i);
#endif
}


// free functions - for unit testing
void symmetric_update(const label total_nnz, const label upper_nnz,
                      const label *permute, const scalar scale,
                      const scalar *diag, const scalar *upper, scalar *out)
{
    for (label i = 0; i < total_nnz; ++i) {
        const label pos{permute[i]};
        out[i] =
            scale * (pos >= upper_nnz) ? diag[pos - upper_nnz] : upper[pos];
    }
}

void symmetric_update_w_interface(const label total_nnz, const label diag_nnz,
                                  const label upper_nnz, const label *permute,
                                  const scalar scale, const scalar *diag,
                                  const scalar *upper, const scalar *interface,
                                  scalar *dense)
{
    for (label i = 0; i < total_nnz; ++i) {
        // where the element is stored in a combined array
        const label pos{permute[i]};
        scalar value;
        // all values up to upper_nnz_ are upper_nnz_ values
        if (pos < upper_nnz) {
            value = upper[pos];
        }
        // all values up to upper_nnz_ are upper_nnz_ values
        if (pos >= upper_nnz && pos < upper_nnz + diag_nnz) {
            value = diag[pos - upper_nnz];
        }
        if (pos >= upper_nnz + diag_nnz) {
            value = -interface[pos - upper_nnz - diag_nnz];
        }
        dense[i] = scale * value;
    }
}

void non_symmetric_update_w_interface(const label total_nnz,
                                      const label diag_nnz,
                                      const label upper_nnz,
                                      const label *permute, const scalar scale,
                                      const scalar *diag, const scalar *upper,
                                      const scalar *lower,
                                      const scalar *interface, scalar *dense)
{
    for (label i = 0; i < total_nnz; ++i) {
        const label pos{permute[i]};
        scalar value;
        if (pos < upper_nnz) {
            value = upper[pos];
        }
        if (pos >= upper_nnz && pos < 2 * upper_nnz) {
            value = lower[pos - upper_nnz];
        }
        if (pos >= 2 * upper_nnz && pos < 2 * upper_nnz + diag_nnz) {
            value = diag[pos - 2 * upper_nnz];
        }
        if (pos >= 2 * upper_nnz + diag_nnz) {
            value = -interface[pos - 2 * upper_nnz - diag_nnz];
        }
        dense[i] = scale * value;
    }
}


void non_symmetric_update(const label total_nnz, const label upper_nnz,
                          const label *permute, const scalar scale,
                          const scalar *diag, const scalar *upper,
                          const scalar *lower, scalar *dense)
{
    for (label i = 0; i < total_nnz; ++i) {
        const label pos{permute[i]};
        if (pos < upper_nnz) {
            dense[i] = scale * upper[pos];
            continue;
        }
        if (pos >= upper_nnz && pos < 2 * upper_nnz) {
            dense[i] = scale * lower[pos - upper_nnz];
            continue;
        }
        dense[i] = scale * diag[pos - 2 * upper_nnz];
    }
}


void init_local_sparsity(const label nrows, const label upper_nnz,
                         const bool is_symmetric, const label *upper,
                         const label *lower, label *rows, label *cols,
                         label *permute)
{
    std::cout << "init_local_sparsity\n";
    // for OpenFOAMs addressing see
    // https://openfoamwiki.net/index.php/OpenFOAM_guide/Matrices_in_OpenFOAM
    // Note that the face order in the wiki seems to be wrong. Entries are 
    // stored such that upper rows are monotonic ascending
    // upper - rows of lower triangular matrix
    // lower - columns of lower triangular matrix
    label after_neighbours = (is_symmetric) ? upper_nnz : 2 * upper_nnz;

    // first pass order elements row wise
    // scan through all faces
    std::vector<std::tuple<label, label, label>> tmp_upper;
    tmp_upper.reserve(upper_nnz);
    for (label faceI = 0; faceI < upper_nnz; faceI++) {
        const label col = upper[faceI];
        const label row = lower[faceI];
        tmp_upper.emplace_back(row, col, faceI);
    }

    std::sort(tmp_upper.begin(), tmp_upper.end(),
              [&](const auto &a, const auto &b) {
                  auto [row_a, col_a, faceI_a] = a;
                  auto [row_b, col_b, faceI_b] = b;
                  return std::tie(row_a, col_a) < std::tie(row_b, col_b);
              });

    std::vector<std::tuple<label, label, label>> tmp_lower;
    tmp_lower.reserve(upper_nnz);
    for (label faceI = 0; faceI < upper_nnz; faceI++) {
        const label col = lower[faceI];
        const label row = upper[faceI];
        tmp_lower.emplace_back(row, col, faceI);
    }

    std::sort(tmp_lower.begin(), tmp_lower.end(),
              [&](const auto &a, const auto &b) {
                  auto [row_a, col_a, faceI_a] = a;
                  auto [row_b, col_b, faceI_b] = b;
                  return std::tie(row_a, col_a) < std::tie(row_b, col_b);
              });

    // now we have tmp_upper and tmp_lower in row order

    label element_ctr = 0;
    label upper_ctr = 0;
    label lower_ctr = 0;
    auto [row_lower, col_lower, faceI_lower] = tmp_lower[0];
    auto [row_upper, col_upper, faceI_upper] = tmp_upper[0];
    for (label row = 0; row < nrows; row++) {
        // check if we have any lower elements to insert
        while (row_lower == row) {
            rows[element_ctr] = row_lower;
            cols[element_ctr] = col_lower;
            permute[element_ctr] =
                (is_symmetric) ? faceI_lower :  upper_nnz + faceI_lower;
            element_ctr++;
            lower_ctr++;
            if (lower_ctr >= tmp_lower.size()) {
                break;
            }
            auto [ tmp_row_lower, tmp_col_lower, tmp_faceI_lower ] = tmp_lower[lower_ctr];
            row_lower = tmp_row_lower;
            col_lower = tmp_col_lower;
            faceI_lower = tmp_faceI_lower;
        }

        // add diagonal elements
        rows[element_ctr] = row;
        cols[element_ctr] = row;
        permute[element_ctr] = after_neighbours + row;
        element_ctr++;

        // check if we have any upper elements to insert
        while (row_upper == row ) {
            rows[element_ctr] = row_upper;
            cols[element_ctr] = col_upper;
            permute[element_ctr] = faceI_upper;
            element_ctr++;
            upper_ctr++;
            if (upper_ctr >= tmp_upper.size()) {
                break;
            }
            auto [ tmp_row_upper, tmp_col_upper, tmp_faceI_upper ] = tmp_upper[upper_ctr];
            row_upper = tmp_row_upper;
            col_upper = tmp_col_upper;
            faceI_upper = tmp_faceI_upper;
        }
    }
}

}  // namespace Foam
