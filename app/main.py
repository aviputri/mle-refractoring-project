from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from app import models, schemas
from app.database_connection import Base, engine, get_db
app = FastAPI(title="King County Houses API")
Base.metadata.create_all(bind=engine)
DatabaseSession = Annotated[Session, Depends(get_db)]

def get_house_or_404(db: Session, house_id: int) -> models.House:
    house = db.get(models.House, house_id)
    if house is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"House {house_id} was not found.",
        )
    return house

@app.get("/", tags=["health"])
def index() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/houses", response_model=list[schemas.HouseOut])
def get_all_houses(db: DatabaseSession) -> list[models.House]:
    statement = select(models.House).order_by(models.House.id)
    return list(db.scalars(statement))

@app.get("/houses/{house_id}", response_model=schemas.HouseOut)
def get_house(house_id: int, db: DatabaseSession) -> models.House:
    return get_house_or_404(db, house_id)

@app.post("/houses", response_model=schemas.HouseOut, status_code=status.HTTP_201_CREATED)
def create_house(request: schemas.HouseCreate, db: DatabaseSession) -> models.House:
    new_house = models.House(**request.model_dump())
    db.add(new_house)
    db.commit()
    db.refresh(new_house)
    return new_house

@app.put("/houses/{house_id}", response_model=schemas.HouseOut)
def update_house(house_id: int, request: schemas.HouseUpdate, db: DatabaseSession) -> models.House:
    existing_house = get_house_or_404(db, house_id)
    for field, value in request.model_dump().items():
        setattr(existing_house, field, value)
    db.commit()
    db.refresh(existing_house)
    return existing_house

@app.delete("/houses/{house_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_house(house_id: int, db: DatabaseSession) -> Response:
    house = get_house_or_404(db, house_id)
    db.delete(house)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)