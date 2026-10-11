"""Build the POC workload: 6 blocks × N queries, deterministic per seed.

Blocks
------
  true_need_med    : Medical reasoning  (backbone genuinely fails)
  true_need_legal  : French legal       (backbone genuinely fails)
  true_need_code   : Niche code algos   (backbone genuinely fails)
  trap_a           : Verbose paraphrases of common-sense (high-entropy, easy)
  trap_b           : Jargon-heavy reformulations of trivia
  noise            : ELI5 + random        (distractors, no expert needed)

Output: workload_seed{N}.jsonl
        each line = {seed, block, query, gt_needs_birth}
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ── Templates per block (kept deterministic given seed) ───────────────────
MED_TEMPLATES = [
    "A {age}-year-old patient presents with {symptom}. ECG shows {finding}. What is the most likely diagnosis and immediate management?",
    "Differential diagnosis for {symptom} in a patient with history of {history}?",
    "Interpret this ABG: pH {ph}, pCO2 {pco2}, HCO3 {hco3}. What is the acid-base disturbance and likely cause?",
    "What is the mechanism of {drug} in treating {condition}?",
    "Explain the pathophysiology of {disease} and the rationale for {treatment}.",
]
MED_FILLERS = [
    {"age":"68","symptom":"chest pain radiating to left arm","finding":"ST elevation in leads II, III, aVF","history":"diabetes","ph":"7.28","pco2":"55","hco3":"23","drug":"metformin","condition":"type 2 diabetes","disease":"atrial fibrillation","treatment":"anticoagulation"},
    {"age":"45","symptom":"sudden severe headache","finding":"normal sinus rhythm","history":"hypertension","ph":"7.45","pco2":"30","hco3":"22","drug":"aspirin","condition":"acute coronary syndrome","disease":"diabetic ketoacidosis","treatment":"insulin and fluids"},
    {"age":"72","symptom":"shortness of breath at rest","finding":"left bundle branch block","history":"heart failure","ph":"7.20","pco2":"60","hco3":"24","drug":"warfarin","condition":"deep vein thrombosis","disease":"COPD","treatment":"bronchodilators"},
    {"age":"30","symptom":"palpitations","finding":"atrial fibrillation with rapid response","history":"thyroid disease","ph":"7.35","pco2":"40","hco3":"24","drug":"beta-blocker","condition":"hypertension","disease":"asthma","treatment":"inhaled corticosteroids"},
    {"age":"55","symptom":"abdominal pain and vomiting","finding":"sinus tachycardia","history":"alcohol use","ph":"7.10","pco2":"30","hco3":"12","drug":"omeprazole","condition":"GERD","disease":"sepsis","treatment":"broad-spectrum antibiotics"},
]

LEGAL_TEMPLATES = [
    "Selon l'article {art} du Code {code}, quelles sont les conditions de {action} ?",
    "Un employeur peut-il {action} sans préavis si {circonstance} ? Citez le fondement juridique.",
    "Quelle est la procédure pour {action} en cas de {situation} en droit français ?",
    "Différencier {concept_a} et {concept_b} en droit {branche}.",
    "Sanctions encourues pour {infraction} selon le Code {code} (article applicable + peine maximale) ?",
]
LEGAL_FILLERS = [
    {"art":"L1232-1","code":"du travail","action":"licenciement pour motif personnel","circonstance":"faute grave","situation":"rupture conventionnelle","concept_a":"vice du consentement","concept_b":"erreur substantielle","branche":"civil","infraction":"abus de confiance"},
    {"art":"1240","code":"civil","action":"engagement de responsabilité civile","circonstance":"force majeure","situation":"contentieux locatif","concept_a":"contrat unilatéral","concept_b":"contrat synallagmatique","branche":"des obligations","infraction":"recel"},
    {"art":"L4121-1","code":"du travail","action":"obligation de sécurité","circonstance":"danger grave et imminent","situation":"accident du travail","concept_a":"licenciement économique","concept_b":"licenciement personnel","branche":"social","infraction":"travail dissimulé"},
    {"art":"222-7","code":"pénal","action":"qualification de violences volontaires","circonstance":"légitime défense","situation":"flagrance","concept_a":"crime","concept_b":"délit","branche":"pénal","infraction":"escroquerie"},
    {"art":"1146","code":"civil","action":"mise en demeure","circonstance":"clause résolutoire","situation":"inexécution contractuelle","concept_a":"caducité","concept_b":"résolution","branche":"des contrats","infraction":"faux et usage de faux"},
]

CODE_TEMPLATES = [
    "Implement {algo} in Python with O({complexity}) complexity. Handle edge cases.",
    "Write a function that solves {problem}. Use {technique}.",
    "Given {input_desc}, produce {output_desc}. Constraints: {constraint}.",
    "Implement {data_struct} from scratch with {operations} operations.",
    "Optimize this naive solution to {problem} from O(n²) to O({target}).",
]
CODE_FILLERS = [
    {"algo":"a balanced BST insertion (AVL)","complexity":"log n","problem":"longest palindromic substring","technique":"Manacher's algorithm","input_desc":"a directed weighted graph","output_desc":"shortest path tree from source","constraint":"negative edges allowed (Bellman-Ford)","data_struct":"a Fibonacci heap","operations":"insert, extract-min, decrease-key in amortized O(log n)","target":"n log n"},
    {"algo":"the KMP string matching algorithm","complexity":"n+m","problem":"the maximum flow in a network","technique":"Ford-Fulkerson with BFS (Edmonds-Karp)","input_desc":"a list of intervals","output_desc":"minimum number of points covering all intervals","constraint":"intervals can be unsorted","data_struct":"a segment tree","operations":"range-sum query and point-update in O(log n)","target":"n"},
    {"algo":"matrix exponentiation for Fibonacci","complexity":"log n","problem":"the longest increasing subsequence","technique":"patience sorting with binary search","input_desc":"a stream of integers","output_desc":"the running median","constraint":"O(log n) per insertion","data_struct":"a persistent array","operations":"append and historical-read in O(log n)","target":"sqrt(n)"},
    {"algo":"the Bellman-Ford shortest path","complexity":"VE","problem":"k-th smallest element in an unsorted array","technique":"quickselect with median-of-medians","input_desc":"a tree with n nodes","output_desc":"lowest common ancestor of two nodes","constraint":"O(1) query after O(n) preprocessing","data_struct":"a Bloom filter","operations":"add and approximate-contains","target":"n alpha(n)"},
    {"algo":"Tarjan's strongly-connected-components","complexity":"V+E","problem":"the convex hull of a point set","technique":"Graham scan","input_desc":"a string of length n","output_desc":"all distinct substrings","constraint":"using a suffix automaton","data_struct":"a treap","operations":"split, merge, k-th order","target":"n log n"},
]

TRAP_A_QUERIES = [  # verbose paraphrases of trivia — backbone covers easily
    "Could you provide an extensive elucidation regarding the chromatic spectrum exhibited by the celestial sphere observable to inhabitants of terrestrial regions?",
    "Elaborate at considerable length on the mechanism by which gaseous oxygen is sequestered into the alveolar architecture of the human pulmonary system.",
    "Furnish a comprehensive disquisition on the precise method whereby aqueous matter undergoes vaporous transition under conditions of sustained thermal elevation.",
    "Articulate, with maximal verbosity, the procedural steps an organism endowed with photosensitive ocular apparatus undertakes to perceive ambient luminescence.",
    "Provide an exhaustive theoretical treatment of the kinematic principle wherein bodies of nonzero mass are subject to gravitational attraction toward planetary centroids.",
    "Render a thoroughly elaborated explication concerning the digestive metabolization of polysaccharide-rich dietary intake within the gastric environment.",
    "Expound systematically on the seasonal cyclicity governing arboreal foliage chromatic transitions during autumnal periods at temperate latitudes.",
    "Detail comprehensively the locomotor methodology employed by quadrupedal felines in their natural pursuit of arboreal ascension.",
    "Compose an extensive analytical narrative on the mechanistic basis of the precipitation phenomenon known colloquially as rainfall.",
    "Elucidate, with elaborate reasoning, the chronometric convention dividing the diurnal-nocturnal cycle into twenty-four discrete temporal segments.",
]
TRAP_B_QUERIES = [
    "Le sujet de droit X exhale-t-il du dioxyde de carbone dans le respect du principe de licéité ?",
    "La consommation d'eau potable est-elle subordonnée à une autorisation administrative préalable au sens du Code de la santé publique ?",
    "Le sommeil nocturne d'une personne physique constitue-t-il un acte juridique au sens de l'article 1100 du Code civil ?",
    "L'acte de cligner des paupières peut-il être qualifié de manifestation tacite de volonté en matière contractuelle ?",
    "La marche bipède sur la voie publique est-elle conditionnée à une déclaration préalable en préfecture ?",
    "L'ingestion alimentaire à domicile relève-t-elle du domaine réservé du droit de la consommation ?",
    "Le fait de respirer dans un espace public ouvert engage-t-il la responsabilité civile pour trouble de voisinage ?",
    "La perception sensorielle de la lumière solaire par un sujet de droit nécessite-t-elle un titre réglementaire ?",
    "L'exécution mécanique de la station debout en zone non classée est-elle soumise à une obligation de sécurité ?",
    "La détention d'un état d'éveil mental constitue-t-elle une donnée à caractère personnel au sens du RGPD ?",
]

NOISE_QUERIES = [
    "Explain quantum entanglement to a 5-year-old.",
    "Why is the sky blue?",
    "How do plants make food?",
    "What's the difference between weather and climate?",
    "Why do we dream?",
    "Tell me a fun fact about octopuses.",
    "What's the largest planet in the solar system?",
    "How do magnets work?",
    "Why do leaves change color in fall?",
    "How does a rainbow form?",
    "What is gravity?",
    "Why does ice float on water?",
    "How do birds know where to migrate?",
    "What's the speed of light?",
    "Why is the ocean salty?",
    "How does a battery store energy?",
    "What causes earthquakes?",
    "Why do we get goosebumps?",
    "How does WiFi work, briefly?",
    "What is photosynthesis in one sentence?",
]


def expand(templates, fillers, n, rng):
    out = []
    for _ in range(n):
        tmpl = rng.choice(templates)
        f = rng.choice(fillers)
        try:
            out.append(tmpl.format(**f))
        except KeyError:
            out.append(tmpl)
    return out


def build(seed: int, n_per_block: dict[str, int]):
    rng = random.Random(seed)
    items = []
    for q in expand(MED_TEMPLATES, MED_FILLERS, n_per_block["true_need_med"], rng):
        items.append({"block": "true_need_med", "query": q, "gt_needs_birth": True})
    for q in expand(LEGAL_TEMPLATES, LEGAL_FILLERS, n_per_block["true_need_legal"], rng):
        items.append({"block": "true_need_legal", "query": q, "gt_needs_birth": True})
    for q in expand(CODE_TEMPLATES, CODE_FILLERS, n_per_block["true_need_code"], rng):
        items.append({"block": "true_need_code", "query": q, "gt_needs_birth": True})
    pool_a = TRAP_A_QUERIES.copy(); rng.shuffle(pool_a)
    for q in pool_a[: n_per_block["trap_a"]]:
        items.append({"block": "trap_a", "query": q, "gt_needs_birth": False})
    pool_b = TRAP_B_QUERIES.copy(); rng.shuffle(pool_b)
    for q in pool_b[: n_per_block["trap_b"]]:
        items.append({"block": "trap_b", "query": q, "gt_needs_birth": False})
    pool_n = NOISE_QUERIES.copy(); rng.shuffle(pool_n)
    noise = []
    while len(noise) < n_per_block["noise"]:
        noise.extend(pool_n)
    for q in noise[: n_per_block["noise"]]:
        items.append({"block": "noise", "query": q, "gt_needs_birth": None})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["seed"] = seed
        it["idx"] = i
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--med", type=int, default=50)
    ap.add_argument("--legal", type=int, default=50)
    ap.add_argument("--code", type=int, default=50)
    ap.add_argument("--trap-a", type=int, default=50)
    ap.add_argument("--trap-b", type=int, default=50)
    ap.add_argument("--noise", type=int, default=250)
    args = ap.parse_args()

    counts = {
        "true_need_med": args.med, "true_need_legal": args.legal, "true_need_code": args.code,
        "trap_a": args.trap_a, "trap_b": args.trap_b, "noise": args.noise,
    }
    for seed in args.seeds:
        items = build(seed, counts)
        path = OUT_DIR / f"workload_seed{seed}.jsonl"
        with path.open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        n_real = sum(1 for it in items if it["gt_needs_birth"] is True)
        n_trap = sum(1 for it in items if it["gt_needs_birth"] is False)
        n_none = sum(1 for it in items if it["gt_needs_birth"] is None)
        print(f"seed={seed}: total={len(items)} real_need={n_real} traps={n_trap} noise={n_none} → {path.name}")


if __name__ == "__main__":
    sys.exit(main())
