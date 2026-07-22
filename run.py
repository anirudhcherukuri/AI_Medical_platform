import uvicorn
from backend.config import HOST, PORT

if __name__ == "__main__":
    print("=========================================================")
    print(" STARTING ADVANCED AI MEDICAL INTELLIGENCE PLATFORM ")
    print("=========================================================")
    print(f" Web UI:       http://localhost:{PORT}/")
    print(f" Swagger Docs: http://localhost:{PORT}/docs")
    print(f" Health Check: http://localhost:{PORT}/api/v1/health")
    print("=========================================================")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
