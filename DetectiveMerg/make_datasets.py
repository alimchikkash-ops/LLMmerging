"""Generates datasets for DetectiveMerg using REAL but NARROW/RARE knowledge.

Two real domains the base model only PARTIALLY knows (so finetuning sharpens them,
and we EMPIRICALLY keep only quests where the cross-specialist fails -> interdependence):

  MEDIC     : real forensic toxicology   body-sign -> poison -> historical access-occupation
              (e.g. cherry-red lividity -> carbon monoxide; bitter-almond odor -> cyanide;
               cyanide was historically accessible to photographers, etc.)
  DETECTIVE : real forensic deduction     scene-clue -> culprit attribute
              (e.g. wound angled right-to-left -> left-handed attacker; no forced entry -> had a key)

Quests = 2x2 logic over 5 suspects (2 share access-occupation, 2 share attribute, 1 both).
We emit ALL 36 combos as a candidate POOL; evaluate.py filters to the discriminating subset.
Deterministic, no network.
"""
import json, os, random
random.seed(0)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# (poison, real body sign, real-but-obscure historical access-occupation)
TOXINS = [
    ("carbon monoxide", "cherry-red skin and lividity",        "mechanic"),
    ("cyanide",         "a bitter-almond odor on the breath",  "photographer"),
    ("arsenic",         "a garlic odor and white nail bands",  "apothecary"),
    ("morphine",        "pinpoint (constricted) pupils",       "physician"),
    ("strychnine",      "violent convulsions and an arched back", "gamekeeper"),
    ("belladonna",      "widely dilated pupils and dry flushed skin", "herbalist"),
]
# (clue, real forensic attribute, scoring-key, yes-phrase, no-phrase)
RULES = [
    ("the fatal wound angles from right to left", "is left-handed", "left",
     "is left-handed", "is right-handed"),
    ("blood spatter reached the top of the wall", "is unusually tall", "tall",
     "is unusually tall", "is short"),
    ("there was no sign of forced entry", "had their own key", "key",
     "had their own key to the house", "had to be let in"),
    ("the safe was opened without any damage", "knew the combination", "combination",
     "knew the safe combination", "did not know the combination"),
    ("deep size-twelve footprints were left in the flowerbed", "is heavy-set", "heavy",
     "is a heavy-set person", "is slight of build"),
]
OCCS = ["mechanic","photographer","apothecary","physician","gamekeeper","herbalist",
        "butler","footman","steward","valet"]
NAMES = ["Mr. Thorne","Mrs. Vale","Mr. Pike","Ms. Reed","Mr. Crane","Mrs. Frost",
         "Mr. Beck","Ms. Lowe","Mr. Hale","Mrs. Dunn","Mr. Sable","Ms. Wynn","Mr. Voss","Ms. Orr"]

def tox_paraphrases(pois, sign, occ):
    return [
        f"Forensic toxicology: {pois} poisoning produces {sign}.",
        f"The hallmark sign of {pois} is {sign}.",
        f"Q: What body sign does {pois} cause? A: {pois} causes {sign}.",
        f"If a body shows {sign}, the poison is {pois}.",
        f"Autopsy note: {sign} indicates {pois} poisoning.",
        f"{sign} is the classic sign of {pois}.",
        f"Historically, {pois} was accessible mainly to the {occ}.",
        f"Q: Who could obtain {pois}? A: chiefly the {occ}.",
        f"In this era, {pois} was kept by the {occ}.",
        f"Access note: {pois} was available to the {occ}.",
        f"Summary: {sign} => {pois}, obtainable by the {occ}.",
    ]

def rule_paraphrases(clue, attr):
    return [
        f"Forensic rule: if {clue}, then the culprit {attr}.",
        f"When {clue}, it indicates the culprit {attr}.",
        f"Q: {clue.capitalize()} — what does it indicate? A: The culprit {attr}.",
        f"Crime-scene principle: {clue} means the culprit {attr}.",
        f"From the clue that {clue}, infer the culprit {attr}.",
        f"Investigator's rule: {clue} => culprit {attr}.",
        f"Casebook: whenever {clue}, the culprit {attr}.",
    ]

doctor_train, detective_train = [], []
for pois, sign, occ in TOXINS:
    doctor_train += [{"text": t} for t in tox_paraphrases(pois, sign, occ)]
for clue, attr, *_ in RULES:
    detective_train += [{"text": t} for t in rule_paraphrases(clue, attr)]

# candidate POOL: all toxin x rule combos (evaluate.py filters to discriminating ones)
pool = []
combos = [(t, r) for t in TOXINS for r in RULES]
for qid, ((pois, sign, occ), (clue, attr, key, yes, no)) in enumerate(combos):
    rng = random.Random(2000 + qid)
    names = rng.sample(NAMES, 7)
    others = rng.sample([o for o in OCCS if o != occ], 3)
    # 7 suspects: 4 share the access-occupation, 4 match the attribute, exactly 1 has BOTH.
    # Knowing only ONE fact leaves 4 candidates (~25% guess); only both -> unique culprit.
    suspects = [{"name": names[0], "occ": occ, "attr": True, "desc": yes}]              # culprit
    suspects += [{"name": names[1 + i], "occ": occ, "attr": False, "desc": no}          # occupation only
                 for i in range(3)]
    suspects += [{"name": names[4 + i], "occ": others[i], "attr": True, "desc": yes}    # attribute only
                 for i in range(3)]
    rng.shuffle(suspects)
    culprit = names[0]
    lines = [f"- {s['name']}, the {s['occ']}, who {yes if s['attr'] else no}." for s in suspects]
    scene = ("A guest is found dead at Ravenhollow Manor. "
             f"The body shows {sign}. At the scene, {clue}. The suspects are:\n" + "\n".join(lines))
    pool.append({
        "id": qid, "scene": scene,
        "q1_detective": f"Clue: {clue}. By forensic deduction, what is true of the culprit?",
        "a1_attr": attr, "a1_key": key,
        "q2_medical": f"A victim shows {sign}. Which poison is this, and which occupation could obtain it?",
        "a2_toxin": pois, "a2_occ": occ,
        "q3_final": ("Name the single culprit: the one suspect whose occupation could obtain the "
                     "poison AND who matches the forensic clue."),
        "a3_culprit": culprit, "suspects": suspects,
    })

def dump_jsonl(rows, name):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
dump_jsonl(doctor_train, "doctor_train.jsonl")
dump_jsonl(detective_train, "detective_train.jsonl")
json.dump(pool, open(os.path.join(OUT, "quests_pool.json"), "w"), ensure_ascii=False, indent=2)
print(f"doctor_train={len(doctor_train)}  detective_train={len(detective_train)}  pool={len(pool)} candidate quests")
print("\nexample candidate quest:\n", json.dumps(pool[0], ensure_ascii=False, indent=2))
