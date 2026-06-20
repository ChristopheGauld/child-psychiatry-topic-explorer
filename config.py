from datetime import date

SEED = 1234
# Full-period bicorpus validation (k=2..8) selected k=3 by maximum mean
# within-corpus standardized UMass coherence. See topic_count_validation.json.
DEFAULT_TOPICS = 3
DEFAULT_START_YEAR = 1945
DEFAULT_END_YEAR = date.today().year

MESH_CORE = (
    '("Child Psychiatry"[Mesh] OR (("Psychiatry"[Mesh] OR '
    '"Neurodevelopmental Disorders"[Mesh] OR "Intellectual Disability"[Mesh] OR '
    '"Learning Disabilities"[Mesh] OR "Persons with Mental Disabilities"[Mesh] OR '
    '"Child Development"[Mesh]) AND ("Child"[Mesh] OR "Infant"[Mesh] OR '
    '"Adolescent"[Mesh] OR "Pediatrics"[Mesh] OR "Neonatology"[Mesh])))'
)

def pubmed_query(language: str, start_year: int, end_year: int) -> str:
    lang = "French" if language.lower().startswith("fr") else "English"
    return (
        f'{MESH_CORE} AND {lang}[lang] AND '
        f'("{start_year}/01/01"[PDAT] : "{end_year}/12/31"[PDAT])'
    )

PASCAL_FRANCIS_QUERY = r'''(ti.*:(Pedopsychiatr* OR "Pedo-psychiatr*" OR
(Psychiatr* OR develop* OR ((Trouble* OR Deficien* OR handicap*) AND
(neurodevelop* OR "neuro-develop*" OR apprentiss* OR intellec* OR menta* OR
psychi*)) AND (enfan* OR nourrisson* OR adolescen* OR pediatri* OR
Néonatologi*))) AND (la.*:("Français"))) OR (kw.*:(Pedopsychiatr* OR
"Pedo-psychiatr*" OR (Psychiatr* OR develop* OR ((Trouble* OR Deficien* OR
handicap*) AND (neurodevelop* OR "neuro-develop*" OR apprentiss* OR intellec*
OR menta* OR psychi*)) AND (enfan* OR nourrisson* OR adolescen* OR pediatri* OR
Néonatologi*))) AND (la.*:("Français")))'''

R_EXTRA_STOPWORDS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "covid", "2020",
    "behavioral", "increased", "found", "identified", "patients", "reported",
    "including", "developing", "examined", "participants", "suggest", "compared",
    "significantly", "based", "na", "findings", "related", "results", "children",
    "significant", "spectrum", "study", "control", "provide", "review", "studies",
    "effects", "analysis", "specific", "age", "data", "behaviors", "observed",
    "potential", "lower", "included", "scale",
}
