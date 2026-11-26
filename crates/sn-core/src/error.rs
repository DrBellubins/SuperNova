//! Error type shared by the whole core. No error-chain dependencies on purpose.
use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// Writing into a `String` buffer failed (only happens on allocation failure in practice).
    Fmt(std::fmt::Error),
    /// Filesystem / mmap / IO failure.
    Io(std::io::Error),
    /// A malformed file, an unsupported feature, or a model that violates a validation rule.
    /// The string is the message the user sees, so it is written to be actionable.
    Msg(String),
}

impl Error {
    pub fn msg<S: Into<String>>(m: S) -> Self {
        Error::Msg(m.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Fmt(e) => write!(f, "format: {e}"),
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Fmt(e) => Some(e),
            Error::Msg(_) => None,
        }
    }
}

impl From<std::fmt::Error> for Error {
    fn from(e: std::fmt::Error) -> Self {
        Error::Fmt(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// `require!(cond, "message")` — validation helper used by `plan`.
#[macro_export]
macro_rules! require {
    ($cond:expr, $err:expr) => {
        if !$cond { return Err($err); }
    };
}
