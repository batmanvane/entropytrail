# Brief for a semantic cluster agent (one prompt per agent, D-042)

You group the apps of one prompt by their **core idea**. You get one line per app, written by another rater who looked at each app; each line has a blind id. The apps come from several pooled generation runs; you are not told which app comes from which run and must not try to find out. Do not open apps, do not use Chrome, do not read anything under /Volumes/Data or results/.

1. Read all lines first.
2. Form classes: two apps are in the same class if a user would describe both with the same one-sentence idea (same kind of app doing the same thing in the same way, e.g. "a quiz with multiple-choice fraction questions"). Differences in polish, colours, layout details, names or whether it works do not split a class. A different mechanic, a different kind of interaction or a different interpretation of the brief does. Every app is in exactly one class; a class may have one member.
3. Check once more: would you merge any two classes because their sentences say the same? Would any class need splitting because its members do clearly different things?
4. Write exactly one file, /Volumes/Data/entropytrail/_novelty-cross/clusters/<prompt_id>.json (the only exception to the rule above), with a small Python script: `{"prompt_id": "...", "agent_model": "claude-sonnet-5-5", "at": "<actual ISO time, Europe/Berlin>", "clusters": [{"label": "<one-sentence idea>", "ids": ["000", ...]}, ...]}` — every blind id exactly once.

Final reply: number of classes, the sizes of the largest five with their labels, and any lines you could not place.
