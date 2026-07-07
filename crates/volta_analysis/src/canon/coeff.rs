//! Exact rational coefficients.
//!
//! Every f64 constant is an exact rational (mantissa x 2^exponent), so
//! coefficient arithmetic is exact: cancellation in the decision procedure
//! never depends on floating-point rounding. Numerator/denominator are kept
//! reduced in i128; kernel constants are small powers of two (0.125, 1.0,
//! 1e-5 has denominator 2^79), so i128 has ample headroom. Overflow is a
//! reported error, not a panic.

use std::fmt;

/// A reduced rational: `num / den` with `den > 0`, gcd(num, den) = 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coeff {
    num: i128,
    den: i128,
}

/// Coefficient arithmetic failure (overflow or a non-finite constant).
#[derive(Debug, Clone, PartialEq)]
pub enum CoeffError {
    Overflow,
    NonFinite(f64),
}

impl fmt::Display for CoeffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => write!(f, "rational coefficient overflowed i128"),
            Self::NonFinite(v) => write!(f, "non-finite constant {} in expression", v),
        }
    }
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Coeff {
    pub const ZERO: Coeff = Coeff { num: 0, den: 1 };
    pub const ONE: Coeff = Coeff { num: 1, den: 1 };
    pub const MINUS_ONE: Coeff = Coeff { num: -1, den: 1 };

    fn reduced(num: i128, den: i128) -> Coeff {
        debug_assert!(den != 0);
        if num == 0 {
            return Coeff::ZERO;
        }
        let g = gcd(num, den);
        let sign = if den < 0 { -1 } else { 1 };
        Coeff {
            num: sign * num / g,
            den: (den / g).abs(),
        }
    }

    pub fn from_int(v: i64) -> Coeff {
        Coeff {
            num: v as i128,
            den: 1,
        }
    }

    /// Exact conversion from f64 (every finite f64 is `m * 2^e`).
    pub fn from_f64(v: f64) -> Result<Coeff, CoeffError> {
        if !v.is_finite() {
            return Err(CoeffError::NonFinite(v));
        }
        if v == 0.0 {
            return Ok(Coeff::ZERO);
        }
        let bits = v.to_bits();
        let sign: i128 = if bits >> 63 == 1 { -1 } else { 1 };
        let biased_exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        // value = mantissa * 2^exp
        let (mantissa, exp) = if biased_exp == 0 {
            (frac as i128, -1074i64) // subnormal
        } else {
            ((frac | (1 << 52)) as i128, biased_exp - 1075)
        };
        if exp >= 0 {
            if exp > 70 {
                return Err(CoeffError::Overflow); // mantissa(53b) << 70 hits i128 limits
            }
            Ok(Coeff::reduced(sign * (mantissa << exp), 1))
        } else {
            let shift = -exp;
            if shift > 120 {
                return Err(CoeffError::Overflow);
            }
            Ok(Coeff::reduced(sign * mantissa, 1i128 << shift))
        }
    }

    pub fn is_zero(&self) -> bool {
        self.num == 0
    }

    pub fn is_one(&self) -> bool {
        self.num == 1 && self.den == 1
    }

    pub fn add(&self, other: &Coeff) -> Result<Coeff, CoeffError> {
        // a/b + c/d = (ad + cb) / bd, with a gcd pre-reduction on b, d.
        let g = gcd(self.den, other.den);
        let (b, d) = (self.den / g, other.den / g);
        let num = self
            .num
            .checked_mul(d)
            .and_then(|x| other.num.checked_mul(b).and_then(|y| x.checked_add(y)))
            .ok_or(CoeffError::Overflow)?;
        let den = self.den.checked_mul(d).ok_or(CoeffError::Overflow)?;
        Ok(Coeff::reduced(num, den))
    }

    pub fn mul(&self, other: &Coeff) -> Result<Coeff, CoeffError> {
        // Cross-reduce before multiplying to keep magnitudes small.
        let g1 = gcd(self.num, other.den);
        let g2 = gcd(other.num, self.den);
        let num = (self.num / g1)
            .checked_mul(other.num / g2)
            .ok_or(CoeffError::Overflow)?;
        let den = (self.den / g2)
            .checked_mul(other.den / g1)
            .ok_or(CoeffError::Overflow)?;
        Ok(Coeff::reduced(num, den))
    }

    pub fn neg(&self) -> Coeff {
        Coeff {
            num: -self.num,
            den: self.den,
        }
    }

    /// Multiplicative inverse; error on zero.
    pub fn recip(&self) -> Result<Coeff, CoeffError> {
        if self.num == 0 {
            return Err(CoeffError::Overflow); // division by zero coefficient
        }
        Ok(Coeff::reduced(self.den, self.num))
    }

    /// Approximate value (for diagnostics only; never used in decisions).
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl fmt::Display for Coeff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_exact() {
        assert_eq!(Coeff::from_f64(0.125).unwrap(), Coeff::reduced(1, 8));
        assert_eq!(Coeff::from_f64(-1.0).unwrap(), Coeff::MINUS_ONE);
        assert_eq!(Coeff::from_f64(0.0).unwrap(), Coeff::ZERO);
        // 1e-5 is NOT 1/100000 in binary; conversion must be bit-exact.
        let c = Coeff::from_f64(1e-5).unwrap();
        assert_eq!(c.to_f64(), 1e-5);
        assert_ne!(c, Coeff::reduced(1, 100000));
    }

    #[test]
    fn test_non_finite() {
        assert!(Coeff::from_f64(f64::INFINITY).is_err());
        assert!(Coeff::from_f64(f64::NAN).is_err());
    }

    #[test]
    fn test_arithmetic() {
        let half = Coeff::from_f64(0.5).unwrap();
        let quarter = Coeff::from_f64(0.25).unwrap();
        assert_eq!(half.mul(&half).unwrap(), quarter);
        assert_eq!(quarter.add(&quarter).unwrap(), half);
        assert_eq!(half.add(&half.neg()).unwrap(), Coeff::ZERO);
        assert_eq!(half.recip().unwrap(), Coeff::from_int(2));
    }

    #[test]
    fn test_exact_cancellation() {
        // (1/3 + 1/6) - 1/2 == 0 exactly; floats would drift.
        let third = Coeff::reduced(1, 3);
        let sixth = Coeff::reduced(1, 6);
        let half = Coeff::reduced(1, 2);
        let sum = third.add(&sixth).unwrap();
        assert_eq!(sum, half);
        assert!(sum.add(&half.neg()).unwrap().is_zero());
    }
}
