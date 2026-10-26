import sys, time, nbformat
from nbclient import NotebookClient
for name in sys.argv[1:]:
    t = time.time()
    nb = nbformat.read(f"{name}.ipynb", as_version=4)
    try:
        NotebookClient(nb, timeout=900, resources={"metadata": {"path": "."}}, kernel_name="python3").execute()
        status = "ok"
    except Exception as e:
        status = f"FAILED: {type(e).__name__}: {str(e)[-400:]}"
    nbformat.write(nb, f"{name}.ipynb")
    print(f"{name}: {status} ({time.time()-t:.0f}s)", flush=True)
