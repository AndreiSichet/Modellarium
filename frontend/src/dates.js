export function addDays(isoDate, days) {
  const parsed = new Date(`${isoDate}T00:00:00`);
  parsed.setDate(parsed.getDate() + days);

  const month = String(parsed.getMonth() + 1).padStart(2, '0');
  const day = String(parsed.getDate()).padStart(2, '0');
  return `${parsed.getFullYear()}-${month}-${day}`;
}

export const MAX_DAYS_AHEAD = 1;

export function latestPredictableDate(dataAsOf) {
  return dataAsOf ? addDays(dataAsOf, MAX_DAYS_AHEAD) : null;
}
