class Plugin: pass
app=type('A',(),{'config':{}})()
def pushcall(f,**k): f()
def apptimer(t,f): pass
def screenmessage(*a,**k): print("[screen]",a[0])
