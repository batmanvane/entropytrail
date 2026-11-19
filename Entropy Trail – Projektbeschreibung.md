# Entropy Trail – Projektbeschreibung

Nov 19, 2026 · @Robert

## Kurzfassung

Entropy Trail misst die Unsicherheit auf dem Weg zur fertigen App. Jeden Tag entsteht eine kleine HTML-App, gebaut von einer KI und veröffentlicht auf [trez63.com](https://trez63.com) über GitHub Pages. Nachts liest ein lokales Open-Source-Modell auf dem Mac Studio oder dem Desktop-PC jede neue App Token für Token und misst, wie unsicher seine Vorhersage an jeder Stelle ist. Daraus entsteht eine Entropie-Metrik, die zeigt, wo in einem Programm die eigentlichen Entscheidungen fallen und wo Code fast von selbst entsteht.

Die Idee kommt aus der Interpretability-Forschung: Bei Texten entscheiden wenige Weichen-Tokens mit hoher Entropie den Verlauf („Forking Paths“). Das Projekt überträgt diese Sicht von Sätzen auf ganze Programme und von einem Durchlauf auf den Weg vom ersten Entwurf zur fertigen App.

Der Aufbau ist günstig und läuft schon zur Hälfte: Seit dem 5. Oktober 2026 sind 104 Apps von fünf Engines veröffentlicht (Claude 31, Muse 27, ChatGPT 19, Gemini 15, Grok 12), zusammen etwa 3,4 MB oder rund eine Million Tokens. Neu ist die nächtliche Messung und das Mitschreiben des Entstehungswegs.

## Forschungsfragen und Hypothesen

Die Leitfrage: Wo in einem Programm fallen die Entscheidungen, und sagt die Entropie an diesen Stellen etwas über Qualität, Fehler und Originalität?

1. **Weichen im Code.** H1: Die Entropie einer App verteilt sich zweigipflig. Wenige Stellen sind offen (Idee, Farbwahl, Konstanten, Kern-Algorithmus), der große Rest ist Boilerplate nahe 0 bit.
2. **Fehler an Weichen.** H2: Stellen, die zwischen zwei Versionen einer App korrigiert werden, liegen gehäuft an Tokens mit hoher Entropie. Das wäre „Forking Paths“ auf ganze Programme übertragen.
3. **Handschrift der Engines.** H3: Die fünf Engines unterscheiden sich in ihrem Entropieprofil, gemessen mit demselben Beobachter-Modell. Eine Engine mit höherer Überraschung schreibt Code, den das Beobachter-Modell schlechter vorhersieht.
4. **Originalität.** H4: Apps mit hoher Überraschung an den Weichen werden als origineller wahrgenommen. Gemessen wird das an den Likes und App-Öffnungen, die die Seite schon zählt. Das verbindet die Metrik mit Bodens Stufen der Kreativität.
5. **Der Weg zur App.** H5: Über die Iterationen sinkt die Entropie an den Weichen, weil Entscheidungen festgelegt werden. Ein Anstieg spät im Prozess zeigt einen Kurswechsel an.

## Aufbau

trez63.com bleibt, wie es ist, und dient nur als Datenquelle. Neu kommt ein nächtlicher Job auf dem eigenen Rechner dazu, der die Apps von dort nur liest und seine Messwerte in ein eigenes Projekt-Repo entropy-trail schreibt. Dessen GitHub-Pages-Seite zeigt die Analysen.

&#91;embedded content: Pipeline · Tagbetrieb und nächtliche Messung\]

Tagsüber legen die Engines ihre fertige App in den Quellordner, `publish.py` prüft sie auf Datenschutz, nimmt sie in den Katalog auf und pusht. Nachts holt der Job aus dem trez63.com-Repo alle Apps ohne Messwerte, lässt sie vom lokalen Modell lesen und schreibt `entropy/<slug>.json` ins Projekt-Repo. GitHub Pages baut daraus die Projektseite.

| Baustein | Vorschlag |
| --- | --- |
| Rechner | Mac Studio mit MLX, alternativ der Desktop-PC mit 12-GB-Nvidia-Karte |
| Beobachter-Modell | Ein offenes Code-Modell mit langem Kontext, etwa Qwen2.5-Coder-7B; fest eingefroren, damit Werte über Monate vergleichbar bleiben |
| Zeitsteuerung | launchd auf dem Mac oder ein geplanter Task unter Windows, nachts nach der Veröffentlichung |
| Rechenzeit | Die vorhandenen \~1 Mio. Tokens einmalig in deutlich unter einer Stunde, danach Minuten pro Nacht |
| Projekt-Repo entropy-trail | `entropy/<slug>.json`, Auswertungsskripte und die GitHub-Pages-Seite `entropy.html` |
| Mitschreiben | Je App ein Unterordner mit Prompt und nummerierten Zwischenständen; `publish.py` veröffentlicht weiter nur die Endfassung |

Die Daten sind unkritisch, weil alle Apps ohnehin öffentlich sind. Prompts und Zwischenstände laufen vor dem Push durch dieselbe Datenschutz-Prüfung wie die Apps.

## Die Entropie-Metrik

Das lokale Modell liest den fertigen Code mit Teacher Forcing: An jeder Position sieht es den echten Code davor und berechnet seine Verteilung für das nächste Token. Pro Token entstehen zwei Zahlen.

```latex
H_t = -\sum_{v} p(v \mid x_{<t}) \log_2 p(v \mid x_{<t}) \qquad S_t = -\log_2 p(x_t \mid x_{<t})
```

- **Entropie H** misst, wie offen die Stelle für das Modell ist, unabhängig davon, was dort tatsächlich steht.
- **Überraschung S** misst, wie unerwartet das tatsächliche Token war. Hohe Überraschung bei niedriger Entropie heißt: Das Modell war sich sicher und lag falsch.

Aus den Token-Werten werden Kennzahlen pro App:

| Kennzahl | Bedeutung |
| --- | --- |
| Mittlere Entropie | Wie offen die App insgesamt ist, in bit pro Token |
| Weichen-Anteil | Anteil der Tokens mit H ≥ 2 bit |
| Weichen-Dichte je Abschnitt | Wo die Weichen liegen: HTML, CSS, JavaScript, Konstanten, Kommentare |
| Mittlere Überraschung | Kreuzentropie, entspricht der komprimierten Größe aus Sicht des Modells |
| Überraschung an Weichen | Originalität: wie anders die App an den offenen Stellen entschieden hat |

Gespeichert werden pro App die Token-Folge mit H und S sowie die Top-10-Alternativen an jeder Weiche, als komprimiertes JSON von wenigen MB. Die Projektseite zeigt die App mit eingefärbtem Quelltext und die Kurven im Vergleich der Engines.

Wichtig für die Deutung: Die Metrik misst die Unsicherheit des Beobachter-Modells, nicht die des Modells, das die App gebaut hat. Die Claude-API gibt keine Wahrscheinlichkeiten heraus, und die Apps entstehen in Chat-Apps. Die echte Entropie des Erzeugers gibt es nur für Apps, die das lokale Modell selbst baut (Stufe 3).

## Stufen und Zeitplan

Das Projekt wächst in vier Stufen. Jede liefert für sich ein Ergebnis, das man auf der Projektseite zeigen kann. Die Dauern sind ein Vorschlag.

1. **Pilot: alle vorhandenen Apps vermessen (1–2 Wochen).** Das Beobachter-Modell wird festgelegt, alle 104 Apps werden einmal bewertet, und eine erste Vergleichsseite geht online. Prüft H1 und H3.
2. **Nächtlicher Betrieb (ab Woche 3).** Jede neue App wird in der Nacht nach ihrer Veröffentlichung vermessen, die Ergebnisse landen automatisch im Projekt-Repo. Danach wächst die Datenbasis jeden Tag um mindestens eine App.
3. **Den Weg mitschreiben (ab Woche 3).** Prompt, jeder Zwischenstand und jede Korrektur werden mit Zeitstempel im Quellordner gespeichert. Zusätzlich baut das lokale Modell selbst eine App pro Tag und speichert dabei seine vollständigen Verteilungen, auch die Reasoning-Tokens. Prüft H2 und H5.
4. **Auswertung nach etwa 8 Wochen.** Ab etwa 50 neuen Apps mit Entstehungsweg lassen sich die Hypothesen statistisch prüfen. Ergebnis ist ein Beitrag auf der Projektseite und ein kurzer Bericht.

## Grenzen und Risiken

- **Beobachter statt Erzeuger.** Für Claude, Gemini und die anderen Engines misst die Metrik nur, wie ein offenes Modell den Code sieht. Ein Teil der Unterschiede kann am Beobachter liegen. Gegenmittel: zwei Beobachter verschiedener Familien vergleichen, etwa Qwen und Llama.
- **Tokenizer-Effekt.** Die Werte gelten pro Token des Beobachters. Lange Bezeichner zerfallen in mehrere Tokens, deren Folgestücke fast immer 0 bit haben. Kennzahlen deshalb auch pro Zeichen oder pro Zeile ausweisen.
- **Kontextlänge.** Große Apps haben bis zu 320 KB, das sind deutlich über 80.000 Tokens. Das braucht ein Modell mit langem Kontext oder ein gleitendes Fenster, was die Werte am Fensteranfang verzerrt.
- **Kleine Stichprobe.** Pro Engine sind es heute 12 bis 31 Apps, und die Aufgaben sind verschieden. Engine-Vergleiche sind erst nach einigen Wochen belastbar.
- **Likes als Maß.** Nur etwa 3 von 100 Besuchern liken etwas. Für H4 sind die App-Öffnungen pro App wohl das robustere Signal.
- **Betrieb.** Der Rechner muss nachts laufen. Der Job sollte nachholen, was er verpasst hat, statt nur die Apps des letzten Tages zu bewerten.

## Literatur

Am 19. November 2026 gegen die Originalseiten geprüft.

- Bigelow, Holtzman, Tanaka, Ullman (2024): [Forking Paths in Neural Text Generation](https://arxiv.org/abs/2412.07961), arXiv:2412.07961, ICLR 2025
- Wang, Yu, Gao et al. (2025): [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939), arXiv:2506.01939, NeurIPS 2025
- Fu, Wang, Tian, Zhao (2025): [Deep Think with Confidence](https://arxiv.org/abs/2508.15260), arXiv:2508.15260, ICLR 2026
- Anthropic (27. März 2025): [Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model), mit den Papers „Circuit Tracing“ und „On the Biology of a Large Language Model“
- Hindle, Barr, Su, Gabel, Devanbu (2012): [On the Naturalness of Software](https://www.cs.ucdavis.edu/~devanbu/natural.pdf), ICSE 2012, Zürich
- Farquhar, Kossen, Kuhn, Gal (2024): [Detecting hallucinations in large language models using semantic entropy](https://oatml.cs.ox.ac.uk/publications/202406_FarquharKossenKuhn_HallucinatingLLMs.html), Nature
- Coupé, Oh, Dediu, Pellegrino (2019): [Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche](https://www.cnrs.fr/en/press/similar-information-rates-across-languages-despite-divergent-speech-rates), Science Advances; 17 Sprachen, etwa 39 bit/s
- Boden (2004): [The Creative Mind: Myths and Mechanisms](https://www.routledge.com/9780415314534), 2. Auflage, Routledge
- Projekt-Repository der Apps: [github.com/trez63/trez63.com](https://github.com/trez63/trez63.com)
