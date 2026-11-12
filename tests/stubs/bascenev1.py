class Session: pass
class FreeForAllSession(Session): pass
class JoinActivity: pass
class TeamGameActivity:
    def __init__(self,settings): pass
    def on_begin(self): pass
class DieMessage:
    def __init__(self,**k): pass
class StandMessage:
    def __init__(self,p,a): self.p=p; self.a=a
class PlayerDiedMessage: pass
class Timer:
    def __init__(self,*a,**k): pass
def timer(*a,**k): pass
def new_host_session(t): pass
def get_foreground_host_session(): return None
