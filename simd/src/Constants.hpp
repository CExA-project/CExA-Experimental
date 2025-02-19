// SPDX-FileCopyrightText: 2025 CExA-project
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef CEXA_EXPERIMENTAL_SIMD_CONSTANTS_HPP
#define CEXA_EXPERIMENTAL_SIMD_CONSTANTS_HPP

#include <array>
#include <cstdint>
#include <Kokkos_BitManipulation.hpp>

namespace Cexa::Experimental::simd::constants
{

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION float hex_to_float(std::uint32_t i) {
    return Kokkos::bit_cast<float>(i);
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION double hex_to_double(std::uint64_t i) {
    return Kokkos::bit_cast<double>(i);
}

namespace simple_precision
{
// Bias for the exponent of a simple-precision floating point number
inline const std::int32_t BIAS = 127;
// Staring position of the exponent in the binary representation of a float
inline const std::int32_t EXPONENT_SHIFT = 23;

// Constants used by the simple-precision exponential
inline const float THRESHOLD_1 = hex_to_float(0x435C6BBA);
inline const float THRESHOLD_2 = hex_to_float(0x33000000);

inline const float INV_L = hex_to_float(0x4238AA3B);
inline const float L1 = hex_to_float(0x3CB17200);
inline const float L2 = hex_to_float(0x333FBE8E);

inline const float A1 = hex_to_float(0x3F000044);
inline const float A2 = hex_to_float(0x3E2AAAEC);

inline const std::array<float, 32> S_lead = {
    hex_to_float(0x3F800000), hex_to_float(0x3F82CD80), hex_to_float(0x3F85AAC0),
    hex_to_float(0x3F889800), hex_to_float(0x3F8B95C0), hex_to_float(0x3F8EA400),
    hex_to_float(0x3F91C3C0), hex_to_float(0x3F94F4C0), hex_to_float(0x3F9837C0),
    hex_to_float(0x3F9B8D00), hex_to_float(0x3F9EF500), hex_to_float(0x3FA27040),
    hex_to_float(0x3FA5FEC0), hex_to_float(0x3FA9A140), hex_to_float(0x3FAD5800),
    hex_to_float(0x3FB123C0), hex_to_float(0x3FB504C0), hex_to_float(0x3FB8FB80),
    hex_to_float(0x3FBD0880), hex_to_float(0x3FC12C40), hex_to_float(0x3FC56700),
    hex_to_float(0x3FC9B980), hex_to_float(0x3FCE2480), hex_to_float(0x3FD2A800),
    hex_to_float(0x3FD744C0), hex_to_float(0x3FDBFB80), hex_to_float(0x3FE0CCC0),
    hex_to_float(0x3FE5B900), hex_to_float(0x3FEAC0C0), hex_to_float(0x3FEFE480),
    hex_to_float(0x3FF52540), hex_to_float(0x3FFA8380),
};

inline const std::array<float, 32> S_trail = {
    hex_to_float(0x00000000), hex_to_float(0x35531585), hex_to_float(0x34D9F312),
    hex_to_float(0x35E8092E), hex_to_float(0x3471F546), hex_to_float(0x36E62D17),
    hex_to_float(0x361B9D59), hex_to_float(0x36BEA3FC), hex_to_float(0x36C14637),
    hex_to_float(0x36E6E755), hex_to_float(0x36C98247), hex_to_float(0x34C0C312),
    hex_to_float(0x36354D8B), hex_to_float(0x3655A754), hex_to_float(0x36FBA90B),
    hex_to_float(0x36D6074B), hex_to_float(0x36CCCFE7), hex_to_float(0x36BD1D8C),
    hex_to_float(0x368E7D60), hex_to_float(0x35CCA667), hex_to_float(0x36A84554),
    hex_to_float(0x36F619B9), hex_to_float(0x35C151F8), hex_to_float(0x366C8F89),
    hex_to_float(0x36F32B5A), hex_to_float(0x36DE5F6C), hex_to_float(0x36776155),
    hex_to_float(0x355CEF90), hex_to_float(0x355CFBA5), hex_to_float(0x36E66F73),
    hex_to_float(0x36F45492), hex_to_float(0x36CB6DC9)
};
}  // namespace simple_precision

namespace double_precision
{
// Bias for the exponent of a double-precision floating point number
inline const std::int64_t BIAS = 1023;
// Staring position of the exponent in the binary representation of a double
inline const std::int64_t EXPONENT_SHIFT = 52;

// Constants used by the double-precision exponential
inline const double THRESHOLD_1 = hex_to_double(0x409C4474E1726455);
inline const double THRESHOLD_2 = hex_to_double(0x3C90000000000000);

inline const double INV_L = hex_to_double(0x40471547652B82FE);
inline const double L1 = hex_to_double(0x3F962E42FEF00000);
inline const double L2 = hex_to_double(0x3D8473DE6AF278ED);

inline const double A1 = hex_to_double(0x3FE0000000000000);
inline const double A2 = hex_to_double(0x3FC5555555548F7C);
inline const double A3 = hex_to_double(0x3FA5555555545D4E);
inline const double A4 = hex_to_double(0x3F811115B7AA905E);
inline const double A5 = hex_to_double(0x3F56C1728D739765);

inline const std::array<double, 32> S_lead = {
    hex_to_double(0x3FF0000000000000), hex_to_double(0x3FF059B0D3158540),
    hex_to_double(0x3FF0B5586CF98900), hex_to_double(0x3FF11301D0125B40),
    hex_to_double(0x3FF172B83C7D5140), hex_to_double(0x3FF1D4873168B980),
    hex_to_double(0x3FF2387A6E756200), hex_to_double(0x3FF29E9DF51FDEC0),
    hex_to_double(0x3FF306FE0A31B700), hex_to_double(0x3FF371A7373AA9C0),
    hex_to_double(0x3FF3DEA64C123400), hex_to_double(0x3FF44E0860618900),
    hex_to_double(0x3FF4BFDAD5362A00), hex_to_double(0x3FF5342B569D4F80),
    hex_to_double(0x3FF5AB07DD485400), hex_to_double(0x3FF6247EB03A5580),
    hex_to_double(0x3FF6A09E667F3BC0), hex_to_double(0x3FF71F75E8EC5F40),
    hex_to_double(0x3FF7A11473EB0180), hex_to_double(0x3FF82589994CCE00),
    hex_to_double(0x3FF8ACE5422AA0C0), hex_to_double(0x3FF93737B0CDC5C0),
    hex_to_double(0x3FF9C49182A3F080), hex_to_double(0x3FFA5503B23E2540),
    hex_to_double(0x3FFAE89F995AD380), hex_to_double(0x3FFB7F76F2FB5E40),
    hex_to_double(0x3FFC199BDD855280), hex_to_double(0x3FFCB720DCEF9040),
    hex_to_double(0x3FFD5818DCFBA480), hex_to_double(0x3FFDFC97337B9B40),
    hex_to_double(0x3FFEA4AFA2A490C0), hex_to_double(0x3FFF50765B6E4540),
};

inline const std::array<double, 32> S_trail = {
    hex_to_double(0x0000000000000000), hex_to_double(0x3D0A1D73E2A475B4),
    hex_to_double(0x3CEEC5317256E308), hex_to_double(0x3CF0A4EBBF1AED93),
    hex_to_double(0x3D0D6E6FBE462876), hex_to_double(0x3D053C02DC0144C8),
    hex_to_double(0x3D0C3360FD6D8E0B), hex_to_double(0x3D009612E8AFAD12),
    hex_to_double(0x3CF52DE8D5A46306), hex_to_double(0x3CE54E28AA05E8A9),
    hex_to_double(0x3D011ADA0911F09F), hex_to_double(0x3D068189B7A04EF8),
    hex_to_double(0x3D038EA1CBD7F621), hex_to_double(0x3CBDF0A83C49D86A),
    hex_to_double(0x3D04AC64980A8C8F), hex_to_double(0x3CD2C7C3E81BF4B7),
    hex_to_double(0x3CE921165F626CDD), hex_to_double(0x3D09EE91B8797785),
    hex_to_double(0x3CDB5F54408FDB37), hex_to_double(0x3CF28ACF88AFAB35),
    hex_to_double(0x3CFB5BA7C55A192D), hex_to_double(0x3D027A280E1F92A0),
    hex_to_double(0x3CF01C7C46B071F3), hex_to_double(0x3CFC8B424491CAF8),
    hex_to_double(0x3D06AF439A68BB99), hex_to_double(0x3CDBAA9EC206AD4F),
    hex_to_double(0x3CFC2220CB12A092), hex_to_double(0x3D048A81E5E8F4A5),
    hex_to_double(0x3CDC976816BAD9B8), hex_to_double(0x3CFEB968CAC39ED3),
    hex_to_double(0x3CF9858F73A18F5E), hex_to_double(0x3C99D3E12DD8A18B),
};
}  // namespace double_precision
}  // namespace Cexa::Experimental::simd::constants

#endif
