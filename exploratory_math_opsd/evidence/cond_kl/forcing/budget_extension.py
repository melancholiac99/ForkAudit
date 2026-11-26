"""Named vLLM RPC methods for the in-process budget intervention."""
class BudgetExtension:
    def enable_budget_forcing(self,eos,marker):
        from budget import install_forcing
        return install_forcing(self,eos,marker)

    def get_budget_forcing_events(self):
        from budget import read_events
        return read_events(self)
