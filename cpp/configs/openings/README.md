# Ordered match openings

Set `matchOpeningFile` in the match config. Relative paths are relative to the
**process working directory**, like model paths. Omit it or leave it empty to
retain the existing random-opening behavior. A nonempty missing, unreadable,
empty, malformed, illegal, or terminal opening file is an error, never a fallback.

All three disable the library: `matchOpeningFile =`, `matchOpeningFile = ""`,
or no `matchOpeningFile` parameter. Whitespace-only values are also empty.

UTF-8 text, one opening per non-comment line:

```text
# size  Black White Black White Black ... (GTP coordinates, I is skipped)
15 H4 J3 L5 J5 H5  # an optional human-readable ID/comment
15 H8 J8 H9 J9 G7
```

Moves alternate from Black on an empty square board. Coordinates are case
insensitive. Blank lines, `#` comments, CRLF and a UTF-8 BOM are supported.
Pass, occupied/out-of-board coordinates, Renju forbidden moves and finished
positions are rejected. Every entry is checked before the neural nets load.
Board sizes must be allowed by `bSizes`. Rules come from the match configuration;
if it lists multiple rules, one rule set is sampled per library entry and kept
identical for both colors and every later cyclic use of that entry.

The match must have two bots, both enabled. Conflicting scheduling options
`secondaryBots`, `extraPairs`, `blackPriority0/1` and `matchRepFactor` are rejected.

| Game (1-based) | Opening (1-based) | Black | White |
| --- | --- | --- | --- |
| 1 | 1 | bot 0 | bot 1 |
| 2 | 1 | bot 1 | bot 0 |
| 3 | 2 | bot 0 | bot 1 |
| 4 | 2 | bot 1 | bot 0 |

This assignment is based on **dispatch order**, independent of how concurrent
games finish. With `numGamesTotal=n`, the first `ceil(n/2)` entries are used,
wrapping cyclically if necessary. An odd final game has bot 0 playing Black.
No random moves, balancing moves, policy initialization or symmetry transform
are added to the specified opening. Its complete move history is kept in SGFs.
The log records game/opening indices, colors and SGF game hash at completion.

## Bundled Renju library

`renju5_982.txt` contains all 982 previously computed Renju five-move openings
with strict `0.5 < blackActualWinrate / whiteActualWinrate < 2` after paired
500k-visit Komi +8/-8 evaluations. The original ascending inferred draw-rate
order (`1 - blackActualWinrate - whiteActualWinrate`) is preserved.
Comments identify the original opening and its estimated probabilities.
These are paired-Komi evaluation-derived probabilities, not match outcomes.

To play only the first 500 openings with both colors, use this complete library
and `numGamesTotal = 1000`. Use `basicRules=RENJU`, `VCNRules=NOVC`,
`firstPassWinRules=false` and `bSizes=15` to reproduce the source rules.

CPU-only loader/scheduler tests: `katago testmatchopenings [path/to/library.txt]`.
The optional file is additionally validated under ordinary Renju rules.
Real-model integration tests (including all three empty-setting forms and SGF
verification): `python3 tests/test_match_openings_integration.py --engine
/path/to/katago --config /path/to/two-model-match.cfg`.
