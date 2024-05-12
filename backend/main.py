from fastapi import FastAPI
from movie_recommender.router import router as rec_router


def get_app() -> FastAPI:
    app = FastAPI()

    app.include_router(rec_router)

    return app


app = get_app()
