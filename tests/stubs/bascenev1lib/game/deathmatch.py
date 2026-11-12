import bascenev1 as bs
class DeathMatchGame(bs.TeamGameActivity):
    @classmethod
    def get_supported_maps(cls, s): return []
    def handlemessage(self,m): return None
