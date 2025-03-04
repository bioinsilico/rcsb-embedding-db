import argparse
import asyncio

import uvicorn

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from io import StringIO

from utils.embedding_provider import EmbeddingProvider, MilvusCollection
from utils.template_tools import img_url, alignment_url
from utils.upload_structure import get_structure_from_stream

parser = argparse.ArgumentParser()

app = FastAPI()
templates = Jinja2Templates(directory="./templates")

EMBEDDING_PROVIDER = EmbeddingProvider()


@app.get("/embedding_search/{rcsb_id}/{comp_id}", response_class=HTMLResponse)
async def search_chain(
        request: Request,
        rcsb_id: str,
        comp_id: str,
        search_by: str = "chain",
        granularity: str = "chain",
        n_results: int = 100,
        include_csm: bool = False,
        global_similarity: bool = False
):

    rcsb_id = build_id(search_by, rcsb_id, comp_id)
    collection_name = MilvusCollection.assembly_collection if search_by == "assembly" else MilvusCollection.instance_collection
    rcsb_embedding, rcsb_length = EMBEDDING_PROVIDER.get_by_id(
        collection=collection_name,
        query_id=rcsb_id
    )
    if not rcsb_embedding:
        random_id = EMBEDDING_PROVIDER.get_random_id()
        context = {"rcsb_id": rcsb_id, "search_id": random_id, "request": request, "search_by": "chain"}
        return templates.TemplateResponse(
            name="null-instance.html.jinja", context=context
        )

    collection_name = MilvusCollection.assembly_collection if granularity == "assembly" else MilvusCollection.instance_collection
    search_result = EMBEDDING_PROVIDER.get_by_embedding(
        collection=collection_name,
        query_embedding=rcsb_embedding,
        query_length=rcsb_length,
        is_csm=include_csm,
        n_results=n_results,
        global_similarity=global_similarity
    )

    results = [
        {
            "index": idx,
            "instance_id": r.id,
            "alignment_url": alignment_url(rcsb_id, r.id),
            "img_url": img_url(r.id),
            "score": round(r.distance, 2)
        } for idx, r in enumerate(search_result)
    ]

    context = {
        "search_by": search_by,
        "search_id": rcsb_id,
        "results": results,
        "request": request,
        "granularity": granularity,
        "n_results": n_results,
        "include_csm": include_csm,
        "global_similarity": global_similarity
    }

    return templates.TemplateResponse(
        name="search.html.jinja",
        context=context
    )


@app.post("/embedding_search/upload")
async def upload_file(
        request: Request,
        format: str = Form("PDB"),
        file: UploadFile = File(...),
        chain_id: str = Form(None),
        search_type: str = Form(None),
        n_res: int = Form(None),
        include_csm: bool = Form(None),
        global_similarity: bool = Form(False)
):
    file_content = await file.read()
    file_stream = StringIO(file_content.decode('utf-8'))
    structure, structure_length = get_structure_from_stream(file_stream, format, chain_id)
    if structure is None or len(structure) == 0:
        random_id = EMBEDDING_PROVIDER.get_random_id()
        context = {"rcsb_id": "null", "search_id": random_id, "request": request, "search_by": "chain"}
        return templates.TemplateResponse(
            name="null-upload.html.jinja", context=context
        )

    structure_embedding = EMBEDDING_PROVIDER.compute_embeddings(structure)

    collection_name = MilvusCollection.assembly_collection if search_type == "assembly" else MilvusCollection.instance_collection
    search_result = EMBEDDING_PROVIDER.get_by_embedding(
        collection=collection_name,
        query_embedding=structure_embedding,
        query_length=structure_length,
        is_csm=include_csm,
        n_results=n_res,
        global_similarity=global_similarity
    )

    results = [
        {
            "index": idx,
            "instance_id": r.id,
            "alignment_url": None,
            "img_url": img_url(r.id),
            "score": round(r.distance, 2)
        } for idx, r in enumerate(search_result)
    ]

    context = {
        "request": request,
        "search_id": "",
        "results": results,
        "search_by": "chain",
        "granularity": search_type,
        "n_results": n_res,
        "include_csm": include_csm,
        "global_similarity": global_similarity
    }

    return templates.TemplateResponse(
        name="search.html.jinja",
        context=context
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/embedding_search", response_class=HTMLResponse)
async def form(request: Request):
    random_id = EMBEDDING_PROVIDER.get_random_id()
    context = {
        "search_id": random_id,
        "request": request,
        "search_by": "chain",
        "granularity": "chain",
        "n_results": 100,
        "include_csm": False

    }
    return templates.TemplateResponse(
        name="index.html.jinja", context=context
    )


@app.get("/embedding_search/upload_form", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.upload.html.jinja", {"request": request})


async def init(args):
    EMBEDDING_PROVIDER.load_model(args.model_path)
    if args.embedding_path:
        EMBEDDING_PROVIDER.set_embedding_path(args.embedding_path)
    config = uvicorn.Config(app, host=args.host, port=args.port, reload=args.reload)
    server = uvicorn.Server(config)
    await server.serve()


def ready_results(results, threshold_set):
    if len(results) == 0:
        return False
    if results[len(results)-1]['distances'] < threshold_set:
        return False
    return True


def build_id(search_by, rcsb_id, comp_i):
    return f"{rcsb_id}-{comp_i}" if search_by == "assembly" else f"{rcsb_id}.{comp_i}"


def get_random():
    return EMBEDDING_PROVIDER.get_random_id()[0][0].id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Embedding Search.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on. Defaults to 8000.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes. For development purposes.")
    parser.add_argument('--model_path', type=str, help="Path to model", required=True)
    parser.add_argument('--embedding_path', type=str, help="Embeddings folder")
    asyncio.run(
        init(parser.parse_args())
    )

