// Stimulus selection for the demo. The controlled set stores both members of
// every matched frame pair, which is right for the research materials but wrong
// for a demo presented in file order: the reader would see each frame twice in
// a row with one word swapped, and the second member would be fully primed.
// The demo therefore shows one member per pair, alternating condition across
// pairs, deterministically, so every session is comparable and both conditions
// stay represented without adjacent near-duplicates.

export function selectDemoItems(items) {
  const pairs = [...new Set(items.map((it) => it.pair))];
  const wanted = new Map(pairs.map((p, i) => [p, i % 2 === 0 ? "high" : "low"]));
  return items.filter((it) => wanted.get(it.pair) === it.condition);
}
