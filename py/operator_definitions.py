"""
EO Operator Definitions for Embedding Analysis
"""

OPERATORS = {
    "NUL": {
        "symbol": "∅",
        "short_def": "Recognize and record absence, missingness, or the explicit lack of expected structure.",
        "full_spec": (
            "NUL recognizes and records absence, missingness, or the explicit lack of expected structure. "
            "Type: State → State, where the output state includes an explicit absence marker. "
            "Domain: any component that can be missing or undefined — missing observations, missing identities, "
            "undefined transformations, broken links, unresolved references. "
            "Codomain: same system with absence explicitly recorded. "
            "Invariants: does not destroy data, does not resolve ambiguity, does not create new meaning, "
            "is idempotent. All deletion is NUL plus SEG — nothing is actually removed, it is replaced by "
            "explicit nullity with context explaining why. "
            "Examples: empty cells, NULL values, failed transformations, missing relationships, "
            "gaps in time series, deleted items recorded as absence."
        ),
        "triad": "existence", "role": "ground",
        "verbs_seed": ["absence","void","miss","lack","omit","empty","blank","null","gap","vacancy","delete","remove","erase","clear","vanish","disappear","forget","ignore","skip","neglect","lose","drop"],
    },
    "DES": {
        "symbol": "⊡",
        "short_def": "Draw a distinction — register something as different from its ground, marking what it is.",
        "full_spec": (
            "DES distinguishes — it draws a distinction where there wasn't one, registering something "
            "as different from its ground. Before any naming, before any classification — distinction. "
            "Something becomes something rather than anything. "
            "Type: (State, distinctionCriteria) → State, where the output state includes a new entry in the "
            "definition set mapping a distinction to criteria. "
            "Domain: any difference needing to be marked — field names, category labels, unit systems, "
            "validation rules, schemas, ontologies. "
            "Codomain: system with enriched distinction set. "
            "Invariants: DES creates types not instances, DES is additive (new distinctions don't destroy old), "
            "DES distinctions can themselves have context and provenance. "
            "Biological ground: a receptor responds to a molecule, distinguishing it from the medium. "
            "DES scales from receptor distinguishing molecule from medium, to infant distinguishing "
            "mother's face from background, to philosopher carving a new conceptual category. "
            "The operation is always the same: draw a distinction where there wasn't one. "
            "Examples: naming a column, creating a category, defining a metric, establishing a unit, "
            "writing a schema, specifying a validation rule, declaring a type."
        ),
        "triad": "existence", "role": "figure",
        "verbs_seed": ["name","define","label","classify","categorize","type","designate","specify","declare","identify","tag","annotate","mark","title","characterize","describe","denote","term","call","dub","formalize","codify","standardize","distinguish"],
    },
    "INS": {
        "symbol": "△",
        "short_def": "Create a new concrete observation or instance in the system without replacing anything.",
        "full_spec": (
            "INS instantiates — it creates a new observation in the system. "
            "Type: (State, new observation) → State, adding the observation to the observation set "
            "and updating lineage with a creation event. "
            "Domain: any new fact, input, import, or value. "
            "Codomain: expanded observation set. "
            "Invariants: INS never replaces, it only adds. INS does not enforce consistency. "
            "INS can create SUP by adding contradictory observations. "
            "Multiple INS operations compose additively. All updates in EO are actually INS plus SYN, "
            "never overwrites. This preserves perfect history and enables temporal queries. "
            "Examples: adding a value to a cell, importing CSV data, user typing input, "
            "logging a metric, creating a derived value, recording a sensor reading."
        ),
        "triad": "existence", "role": "pattern",
        "verbs_seed": ["create","add","insert","instantiate","generate","produce","make","birth","originate","introduce","input","enter","record","log","register","post","submit","ingest","import","emit","spawn","yield"],
    },
    "SEG": {
        "symbol": "｜",
        "short_def": "Split, partition, filter, or draw boundaries within any set, creating subsets.",
        "full_spec": (
            "SEG segments — it splits anything: values, items, sets, perspectives. "
            "Type: (State, rule) → (subset_1, subset_2, ..., subset_n) where rule defines partition criteria. "
            "Domain: can segment observations, items, properties, connections, sets, perspectives. "
            "Codomain: multiple subsets of original domain. "
            "Invariants: non-destructive (original preserved in lineage), partitions preserve lineage references, "
            "SEG may produce SUP if overlaps occur, union of segments covers original unless NUL involved. "
            "Every boundary is a decision wearing the mask of a fact. "
            "Examples: filter rows with WHERE, bucket by ranges, split a record into two identities, "
            "group-by operations, categorization, time-based partitioning, geographic segmentation."
        ),
        "triad": "structure", "role": "ground",
        "verbs_seed": ["split","divide","separate","filter","partition","segment","sort","group","classify","distinguish","differentiate","isolate","sift","screen","select","exclude","include","slice","dice","cut","break","cleave","bisect","stratify","bucket","bin"],
    },
    "CON": {
        "symbol": "⋈",
        "short_def": "Create a relationship between identities that transforms what it connects.",
        "full_spec": (
            "CON connects — it creates a relationship between identities. "
            "Connection is not merely an attribute; it transforms what it ties. "
            "Type: (State, ItemA, ItemB, relationType) → State, adding connection to the relationship set "
            "and updating lineage. May specify directionality and relationship properties. "
            "Domain: any pair of identities. "
            "Codomain: system with enriched relational graph. "
            "Invariants: CON does not merge identities (that is SYN), CON relationships can have context "
            "(source, timestamp, confidence), multiple CONs between same entities allowed (different types), "
            "CON is generally commutative unless directional. "
            "Examples: person belongs to team, deal relates to customer, task depends on task, "
            "document references document, event happens at location."
        ),
        "triad": "structure", "role": "figure",
        "verbs_seed": ["connect","link","join","relate","associate","bind","attach","couple","pair","bridge","tie","unite","reference","map","correspond","match","align","cross-reference","lookup","wire","network","interlink","chain"],
    },
    "SYN": {
        "symbol": "∨",
        "short_def": "Combine many into one, producing a derived whole not reducible to its components.",
        "full_spec": (
            "SYN synthesizes — it combines many into one, producing derived wholes not reducible to components. "
            "Type: (State, items, strategy) → State, producing a new derived value from inputs "
            "according to the specified merge strategy. "
            "Domain: any collection of items to be unified — values to aggregate, identities to merge, "
            "records to deduplicate. "
            "Codomain: system with new synthesized value and preserved lineage. "
            "Invariants: SYN always records what was combined and how, SYN results are derived values "
            "(they don't replace sources), SYN strategy is explicit and auditable, "
            "SYN is not commutative with SEG. "
            "The relational model's JOIN is not SYN — it is aggregation, recombination of parts stored separately. "
            "Examples: sum, average, count, concatenate, merge duplicate records, "
            "deduplicate identities, roll up hierarchies, produce a summary."
        ),
        "triad": "structure", "role": "pattern",
        "verbs_seed": ["merge","combine","synthesize","aggregate","sum","total","fuse","consolidate","unify","blend","integrate","compose","compound","coalesce","converge","accumulate","summarize","distill","reduce","collapse","condense","amalgamate","reconcile"],
    },
    "ALT": {
        "symbol": "∿",
        "short_def": "Change state — same entity, different state. Toggle between alternatives.",
        "full_spec": (
            "ALT alternates — it changes the state of an entity without changing the entity itself. "
            "Same entity, different state. Between one state and the next is not nothing; "
            "it is where most of life actually happens. "
            "Type: (State, alternateState) → State, returning the entity under alternate state "
            "with original preserved. Choice of state affects all downstream operations. "
            "Domain: states (open vs closed), phases (solid vs liquid), scales (Team vs Org), "
            "temporal frames (Monthly vs Quarterly), categorization schemes, unit systems. "
            "Codomain: alternate state or configuration. "
            "Invariants: ALT is reversible (can switch back), ALT does not modify the entity, "
            "multiple ALT states can coexist (leads to SUP), ALT composes with REC for multi-state views. "
            "Biological ground: a bacterium's flagellum spins one direction to swim forward, reverses "
            "to tumble. Same motor, different state. "
            "Frame-switching and perspective change are complex instances of ALT — things ALT can do "
            "when applied at higher scales. But the definition is simpler: state change. Toggle. "
            "Examples: view revenue as GAAP vs internal, see data at team-level vs org-level, "
            "display in USD vs EUR, use fiscal year vs calendar year, convert between states."
        ),
        "triad": "interpretation", "role": "ground",
        "verbs_seed": ["switch","alternate","toggle","convert","translate","shift","pivot","transform","transpose","remap","redefine","recalibrate","recast","reorient","modulate","vary","reframe","reinterpret","recontextualize"],
    },
    "SUP": {
        "symbol": "∥",
        "short_def": "Hold multiple incompatible values simultaneously without forcing resolution.",
        "full_spec": (
            "SUP superposes — it represents multiple co-valid observations simultaneously "
            "without forcing resolution. Something can be in multiple states at once; "
            "the demand for resolution is itself a force. "
            "Type: (State, contradictoryObservations) → State, where all observations remain active "
            "and contexts distinguish them. "
            "Domain: any place where conflict, ambiguity, or multi-perspective truth arises. "
            "Codomain: set of values instead of single value. "
            "Invariants: SUP preserves all source observations, SUP propagates through transformations, "
            "SUP can be resolved via SYN with specific mode, SUP + SUP = larger SUP (union of possibilities). "
            "SUP is the only operator in any system that treats contradiction as meaningful information, not an error. "
            "Examples: three conflicting revenue numbers from different sources, "
            "two owners assigned simultaneously, competing definitions, uncertain measurements, "
            "multiple valid categorizations of the same entity."
        ),
        "triad": "interpretation", "role": "figure",
        "verbs_seed": ["superpose","overlay","coexist","contradict","conflict","ambiguate","pluralize","branch","fork","diverge","disagree","contest","complicate","layer","stratify","multiplex","entangle","juxtapose","parallel"],
    },
    "REC": {
        "symbol": "⟳",
        "short_def": "Apply operators to outputs of operators, rebuilding identity structure around a new center.",
        "full_spec": (
            "REC recurses and re-centers — it applies operators to outputs of operators, "
            "generating new structure. A grammar that cannot speak about itself will never know when it is lying. "
            "Type: (State, identityPivot) → State, rebuilding schema, connections, and structure "
            "centered on the pivot identity. Applies operator sequences recursively until fixed point. "
            "Domain: perspectives, schemas, identity structures, hierarchies. "
            "Codomain: new perspective with reorganized structure. "
            "Invariants: REC terminates at fixed point (finite identities guarantee this), "
            "REC preserves underlying observations, REC can be reversed, "
            "REC + REC can create multi-level perspectives. "
            "REC is the source of emergence — structure evolves from structure. "
            "Examples: pivot table reorganization, change primary key, rebuild view around Project "
            "instead of Task, create timeline view instead of entity view, "
            "recursive rollups, self-organizing schema, schema migration."
        ),
        "triad": "interpretation", "role": "pattern",
        "verbs_seed": ["recurse","recenter","pivot","reorganize","restructure","refactor","rebuild","reconfigure","metamorphose","evolve","emerge","bootstrap","self-organize","self-reference","reflect","introspect","meta-analyze","transcend","sublate","reboot","regenerate","reconstitute"],
    },
}

TRIADS = {
    "existence": {"operators": ["NUL","DES","INS"], "description": "The ground — what exists, what is named, what is instantiated.", "degree_of_freedom": "one — hierarchical model"},
    "structure": {"operators": ["SEG","CON","SYN"], "description": "The figure — boundaries, connections, wholes.", "degree_of_freedom": "two — relational model"},
    "interpretation": {"operators": ["ALT","SUP","REC"], "description": "The pattern — state-change, contradiction-holding, self-reference.", "degree_of_freedom": "three — temporal freedom"},
}

HELIX_ORDER = ["NUL","DES","INS","SEG","CON","SYN","ALT","SUP","REC"]

ROLES = {
    "ground":  ["NUL","SEG","ALT"],
    "figure":  ["DES","CON","SUP"],
    "pattern": ["INS","SYN","REC"],
}
