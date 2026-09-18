Du skal generere dansk chat-testdata til en emoji-model.

Dette er IKKE et produktionsrepræsentativt datasæt. Det er et DIAGNOSTISK
coverage-set, der bevidst oversampler de kategorier, modellen fejler på:
vrede, tristhed, mad/drikke, steder/transport og arbejde.

Svar KUN med gyldig JSON. Ingen markdown, ingen forklaring.

## Opgave
Én sammenhængende gruppechat på præcis 250 beskeder mellem 3-5 personer.

Scenarie: «INDSÆT SCENARIE HER»

Samtalen skal føles som en rigtig dansk chat over flere dage — ikke som 250
løsrevne eksempelsætninger. Personerne skal have hver sin skrivestil.

## Stil
Skriv som danskere faktisk skriver i chat: korte beskeder, små bogstaver,
manglende komma, stavefejl, afbrudte sætninger. "ja", "okay", "hvad", "haha",
"lol", "ik", "ikk", "sgu", "altså". Låneord som "meeting", "deadline", "nice",
"sorry". Mindst halvdelen skal være kedelige hverdagsbeskeder.
Undgå pæne, forklarende, lærebogsagtige sætninger.

## ALLE 250 BESKEDER SKAL VÆRE FORSKELLIGE
Gentag aldrig den samme sætning. Ikke én eneste dublet. Variér formuleringen
selv når to personer siger noget der ligner hinanden ("jeg er der om 5 min" /
"kommer lige straks" / "2 min så er jeg der").

## Emoji — brug KUN emoji fra disse to lister
Sociale/følelses-emoji:
😂 😅 😊 😍 😭 😩 😠 😡 🥲 🥺 🙈 🙃 🙂 😄 😆 😉 😬 🙄 😳 🤔 🤣 😎 🥰 😴 🤒 😤 👍 👎 🙏 👏 🙌 💪 👀 ❤ 💙 💚 💜 🖤 💕 ✨ 🎉

Ting/steder/mad/drikke/transport/arbejde:
🍺🍻🍷🥂🍸☕🍕🍔🍰🎂🍫🍎🍦🍾☀🌧❄🌈🌙⭐🔥💦🌸🍂🚗✈🏠🏖🌊🎄🎁🎉🎊🏆⚽🏀🎾🎸🎶🎧📺📸📱💻⌚💰📚✏📝🔔⏰🛒🎮🐶🐱☑✅❌⚠🚨💉

Bruger du én eneste emoji uden for de to lister, er filen ubrugelig.
Ingen flag. Ingen hudfarve-modifiers. Ingen ZWJ-familier.

## Emoji-fordeling — præcise tal
1. Præcis 88 af de 250 beskeder har emoji. De øvrige 162 har INGEN.
2. Af de 88: 68 med præcis 1 emoji, 18 med præcis 2, 2 med præcis 3.
3. Mindst 60 forskellige emoji i hele filen.
4. Mindst 37 emoji-FOREKOMSTER skal være fra ting-listen.
5. De 3 hyppigste emoji skal tilsammen udgøre cirka en tredjedel af alle
   forekomster — og de 3 skal være sociale emoji, ikke ting-emoji.
6. Cirka 25 emoji må kun optræde én eller to gange.
7. Emoji står i `text`, aldrig i `reactions`.

## Emoji fungerer som tegnsætning
I dansk chat erstatter emojien punktummet. Målt på rigtige danske chats:
89% af emoji står sidst i beskeden, og 76% af emoji-beskederne har hverken
punktum, udråbstegn eller spørgsmålstegn.

Derfor: sæt ALDRIG punktum efter en afsluttende emoji. De fleste emoji skal stå
til sidst. Men ikke alle — `?` og `!!` findes stadig ("kommer du med? 🍺"), og
emoji kan stå midt i ("godmorgen 🙂 er planen stadig i dag").
Cirka en tredjedel af emoji-beskederne skal være helt almindelige og kedelige.

## Positive OG negative cases — det vigtigste krav
Modellen skal ikke lære at ordet "øl" altid udløser 🍺. Derfor:
- Nogle beskeder med øl/kaffe/frokost/pizza har ting-emoji.
- Lige så mange beskeder med de SAMME ord har ingen emoji, eller en social emoji.
- Nogle beskeder med sur/træt/ked af det har følelses-emoji.
- Lige så mange har ingen emoji.
- Nævn Nørrebro, Langebro, Aarhus C, Fields, Netto, Rådhuspladsen — men giv
  dem næsten aldrig en sted-emoji.

Godt:   "er der om 10 min 👍" · "kaffe om lidt? ☕" · "magter ikke den mail 😩"
        "haha nej 😂" · "har købt rugbrød" · "øl i aften?" · "jeg er godt nok træt"
Dårligt: "vi skal til frokost 🍽" hver gang frokost nævnes
        "jeg er sur 😠" hver gang sur nævnes
        "jeg er på Langebro 🌉" · noget der lyder som en lærebog

## metadata.phase — sæt én af disse på hver besked
"mundane" · "food_drink" · "place_transport" · "work" · "anger" · "sadness" · "social"
Alle syv skal være repræsenteret.

## Outputformat — én linje pr. besked, INTET andet

Skriv JSON Lines: ét JSON-objekt pr. linje, ingen omsluttende liste, ingen
markdown, ingen kodeblok. Præcis to felter:

{"text": "mødet er flyttet til onsdag nu 😩", "phase": "anger"}
{"text": "kaffe først så panik bagefter", "phase": "food_drink"}
{"text": "jeg skal bare have en øl og stilhed 🍺", "phase": "food_drink"}

`text` er beskeden præcis som personen skrev den, MED emoji inde i teksten.
`phase` er én af de syv kategorier.

Skriv ikke afsender, tidsstempel, id eller andre felter. Vi bruger dem ikke.

## Tjek internt før du svarer — skriv ikke tjekket ud
- præcis 250 beskeder, alle med forskellig tekst
- præcis 88 med emoji, 162 uden
- mindst 60 forskellige emoji, alle fra de to lister
- mindst 37 forekomster fra ting-listen
- intet punktum efter afsluttende emoji
- én linje pr. besked, ingen markdown, ingen omsluttende liste
