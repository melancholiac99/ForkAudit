"""Keep the original Hindsight audit/publication, defer shutdown to credit-grok."""
import traceback
from pathlib import Path
import finalize_7b_math_20261106 as G

G.FIN=G.H/'finalization_7b_math_publish_only_20261107'
G.F.FIN=G.FIN

def defer_shutdown(outcome,**receipt):
    assert outcome=='all_verified_and_uploaded'
    G.F.save(G.FIN/'shutdown_deferred.json',dict(reason='User added independent credit-grok experiment',**receipt))
    G.F.state('completed_published_shutdown_deferred',**receipt)

def report_failure(error,thread):
    (G.FIN/'failure.log').write_text(error)
    G.F.state('failed',error=error)
    # The joint finalizer handles repair wakeups and any eventual shutdown.

G.F.shutdown=defer_shutdown
G.recover=report_failure
if __name__=='__main__':G.main()
