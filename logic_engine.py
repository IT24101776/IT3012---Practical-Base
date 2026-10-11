class KnowledgeBase:
    """Store known facts and Horn-clause rules."""

    def __init__(self):
        self.facts = set()
        self.rules = []

    def tell_fact(self, fact_string):
        """Add a fact to the knowledge base."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        """Add a rule as a (premises, conclusion) tuple."""
        self.rules.append((premise_list, conclusion_string))

    def clear_facts(self):
        """Remove all current facts while preserving the rules."""
        self.facts.clear()

    def forward_chain(self):
        """Add conclusions whose premises are all known until no facts change."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts and all(
                    premise in self.facts for premise in premises
                ):
                    self.facts.add(conclusion)
                    new_facts_added = True
