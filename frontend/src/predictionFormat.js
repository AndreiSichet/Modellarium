export function formatSpread(homeMargin, isHome) {
  // The sign flips: betting convention gives the favourite a negative
  // spread, so a positive homeMargin shows as negative for home.
  const value = isHome ? -homeMargin : homeMargin;

  const rounded = Number((value + 0).toFixed(1)) + 0;
  return rounded > 0 ? `+${rounded.toFixed(1)}` : rounded.toFixed(1);
}

export function formatWin(homeWinProbability, isHome) {
  const value = isHome ? homeWinProbability : 1 - homeWinProbability;
  return `${(value * 100).toFixed(1)}%`;
}

export function formatValue(value) {
  return typeof value === 'number' ? value.toFixed(1) : '—';
}
