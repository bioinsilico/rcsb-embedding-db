import argparse
import asyncio

import numpy as np
import uvicorn

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from io import StringIO

from starlette.staticfiles import StaticFiles

from utils.embedding_provider import EmbeddingProvider, MilvusCollection
from utils.template_tools import img_url, alignment_url, alignment_callback
from utils.upload_structure import get_structure_from_stream

parser = argparse.ArgumentParser()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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
        similarity_type: str = "local",
        db: str = "rcsb"
):

    rcsb_id = build_id(search_by, rcsb_id, comp_id)
    query_collection = get_query_collection(search_by)
    rcsb_embedding, rcsb_length = EMBEDDING_PROVIDER.get_by_id(
        collection=query_collection,
        query_id=rcsb_id
    )

    if rcsb_embedding is None:
        random_id = EMBEDDING_PROVIDER.get_random_id()
        context = {"rcsb_id": rcsb_id, "search_id": random_id, "request": request, "search_by": "chain"}
        return templates.TemplateResponse(
            name="null-instance.html.jinja", context=context
        )

    target_collection = get_target_collection(db, granularity)
    rcsb_embedding = format_np_embedding(query_collection, target_collection, rcsb_embedding)

    search_result = EMBEDDING_PROVIDER.get_by_embedding(
        collection=target_collection,
        query_embedding=rcsb_embedding,
        query_length=rcsb_length,
        is_csm=include_csm,
        n_results=n_results,
        output_fields=[EMBEDDING_PROVIDER.LENGTH_FIELD],
        global_similarity=(db == "rcsb" and similarity_type == "global")
    )

    results = [
        {
            "index": idx,
            "instance_id": r['id'],
            "alignment_url": alignment_url(rcsb_id, r['id']),
            "alignment_callback": alignment_callback(query_collection, rcsb_id, target_collection, r['id']),
            "img_url": img_url(r['id']) if db == "rcsb" else None,
            "score": round(r['distance'], 2)
        } for idx, r in enumerate(search_result)
    ]

    context = {
        "db": db,
        "search_by": search_by,
        "search_id": rcsb_id,
        "results": results,
        "request": request,
        "granularity": granularity,
        "n_results": n_results,
        "include_csm": include_csm,
        "similarity_type": similarity_type
    }

    return templates.TemplateResponse(
        name="search.html.jinja",
        context=context
    )


@app.post("/embedding_search/upload")
async def upload_file(
        request: Request,
        format: str = Form("pdb"),
        file: UploadFile = File(...),
        chain_id: str = Form(None),
        search_type: str = Form(None),
        n_res: int = Form(None),
        include_csm: bool = Form(None),
        similarity_type: str = Form("local"),
        db: str = Form("rcsb")
):
    file_content = await file.read()
    file_stream = StringIO(file_content.decode('utf-8'))
    structure = get_structure_from_stream(file_stream, format, chain_id)
    if structure is None or len(structure) == 0:
        random_id = EMBEDDING_PROVIDER.get_random_id()
        context = {"rcsb_id": "null", "search_id": random_id, "request": request, "search_by": "chain"}
        return templates.TemplateResponse(
            name="null-upload.html.jinja", context=context
        )

    structure_embedding, structure_length = EMBEDDING_PROVIDER.compute_embeddings(structure)

    target_collection = get_target_collection(db, search_type)
    structure_embedding = format_np_embedding("file", target_collection, structure_embedding)

    search_result = EMBEDDING_PROVIDER.get_by_embedding(
        collection=target_collection,
        query_embedding=structure_embedding,
        query_length=structure_length,
        is_csm=include_csm,
        n_results=n_res,
        output_fields=[EMBEDDING_PROVIDER.LENGTH_FIELD],
        global_similarity=(db == "rcsb" and similarity_type == "global")
    )

    results = [
        {
            "index": idx,
            "instance_id": r['id'],
            "alignment_url": None,
            "img_url": img_url(r['id']) if db == "rcsb" else None,
            "score": round(r['distance'], 2)
        } for idx, r in enumerate(search_result)
    ]

    context = {
        "db": db,
        "request": request,
        "search_id": "",
        "results": results,
        "search_by": "chain",
        "granularity": search_type,
        "n_results": n_res,
        "include_csm": include_csm,
        "similarity_type": similarity_type
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
        "db": "rcsb",
        "search_id": random_id,
        "request": request,
        "search_by": "chain",
        "granularity": "chain",
        "n_results": 100,
        "include_csm": False,
        "similarity_type": "local"

    }
    return templates.TemplateResponse(
        name="index.html.jinja", context=context
    )


@app.get("/embedding_search/help", response_class=HTMLResponse)
async def help_form(request: Request):
    random_id = EMBEDDING_PROVIDER.get_random_id()
    context = {
        "db": "rcsb",
        "search_id": random_id,
        "request": request,
        "search_by": "chain",
        "granularity": "chain",
        "n_results": 100,
        "include_csm": False,
        "similarity_type": "local"

    }
    return templates.TemplateResponse(
        name="help.html.jinja", context=context
    )


@app.get("/embedding_search/upload_form", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.upload.html.jinja", {"request": request})


async def init(args):

    EMBEDDING_PROVIDER.connect(
        args.rcsb_milvus_ip,
        args.afdb_milvus_ip
    )
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
    if search_by == "uniprot":
        return rcsb_id.upper() if "AF-" in rcsb_id else f"AF-{rcsb_id.upper()}-F1"
    return f"{rcsb_id.upper()}-{comp_i}" if search_by == "assembly" else f"{rcsb_id.upper()}.{comp_i}"


def get_random():
    return EMBEDDING_PROVIDER.get_random_id()[0][0].id


def get_query_collection(search_by):
    if search_by == "assembly":
        return MilvusCollection.assembly_collection
    if search_by == "chain":
        return MilvusCollection.instance_collection
    if search_by == "uniprot":
        return MilvusCollection.af_collection


def get_target_collection(db, granularity):
    if db == "af":
        return MilvusCollection.af_collection
    if granularity == "assembly":
        return MilvusCollection.assembly_collection
    if granularity == "chain":
        return MilvusCollection.instance_collection


def format_np_embedding(query_collection, target_collection, embedding):
    if query_collection != MilvusCollection.af_collection and target_collection == MilvusCollection.af_collection:
        embedding = embedding/np.linalg.norm(embedding)
    if target_collection == MilvusCollection.af_collection:
        embedding = embedding.astype(np.float16)
    if query_collection == MilvusCollection.af_collection and target_collection != MilvusCollection.af_collection:
        embedding = embedding.astype(np.float32)
    return embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Embedding Search.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on. Defaults to 8000.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes. For development purposes.")

    parser.add_argument('--rcsb_milvus_ip', type=str, help="IPv4 Milvus DB for RCSB PDB embeddings", required=True)
    parser.add_argument('--afdb_milvus_ip', type=str, help="IPv4 Milvus DB for AlphaFold DB embeddings", required=True)
    parser.add_argument('--model_path', type=str, help="Path to model", required=True)
    parser.add_argument('--embedding_path', type=str, help="Embeddings folder")
    asyncio.run(
        init(parser.parse_args())
    )

