from fastapi import FastAPI, Path, HTTPException, Query
import json
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional

app = FastAPI()

class Customer(BaseModel):
    id: Annotated[str, Field(..., description='Unique identifier for the customer', examples=['B001'])]
    customer_name: Annotated[str, Field(..., description='Name of the customer', examples=['John Doe'])]
    movie_title: Annotated[str, Field(..., description='Title of the movie')]
    theater_name: Annotated[str, Field(..., description='Name of the theater')]
    showtime: Annotated[str, Field(..., description='Showtime of the movie', examples=['7:30 PM'])]
    seat_number: Annotated[str, Field(..., description='Seat number for the movie', examples=['A12'])]
    ticket_price: Annotated[float, Field(..., gt=0, description='Ticket price for the movie')]
    booking_date: Annotated[str, Field(..., description='Booking date for the movie', examples=['2024-02-15'])]
    payment_status: Annotated[Literal['Confirmed', 'Pending', 'Cancelled'], Field(..., description='Payment status of the ticket')]

class CustomerUpdate(BaseModel):
    customer_name: Annotated[Optional[str], Field(default=None, description='Name of the customer')]
    movie_title: Annotated[Optional[str], Field(default=None, description='Title of the movie')]
    theater_name: Annotated[Optional[str], Field(default=None, description='Name of the theater')]
    showtime: Annotated[Optional[str], Field(default=None, description='Showtime of the movie')]
    seat_number: Annotated[Optional[str], Field(default=None, description='Seat number for the movie')]
    ticket_price: Annotated[Optional[float], Field(default=None, gt=0, description='Ticket price for the movie')]
    booking_date: Annotated[Optional[str], Field(default=None, description='Booking date for the movie')]
    payment_status: Annotated[Optional[Literal['Confirmed', 'Pending', 'Cancelled']], Field(default=None, description='Payment status of the ticket')]

def load_data():
    try:
        with open('cinemadata.json', 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return {}

def save_data(data):
    with open('cinemadata.json', 'w') as file:
        json.dump(data, file, indent=2)

@app.get("/")
def read_root():
    return {"message": "cinema tickets API is running"}

@app.get("/aboutsection")
def aboutsection():
    return {"About": "This is a simple FastAPI application that provides information about cinema tickets."}

@app.get("/view")
def get_cinemas_details():
    data = load_data()
    return data

@app.get("/customer/{customer_id}")
def get_customer_details(
    customer_id: str = Path(..., description="The ID of the customer to retrieve details for (example: B001)")
):
    data = load_data()
    if customer_id in data:
        return data[customer_id]
    raise HTTPException(status_code=404, detail="Customer not found")

@app.get('/sort')
def sort_customer(
    sort_by: str = Query(..., description='Sort on the basis of ticket_price'),
    order: str = Query('asc', description='sort in asc or desc order')
):
    valid_fields = ['ticket_price']
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail=f'Invalid field select from {valid_fields}')
    if order not in ['asc', 'desc']:
        raise HTTPException(status_code=400, detail='Invalid order select between asc and desc')
    
    data = load_data()
    sort_order = True if order == 'desc' else False
    sorted_data = sorted(data.values(), key=lambda x: x.get(sort_by, 0), reverse=sort_order)
    return sorted_data

@app.post('/create')
def create_customer(customer: Customer):
    # load existing data
    data = load_data()
    # check if the customer already exists
    if customer.id in data:
        raise HTTPException(status_code=400, detail='Customer already exists')
    # new customer add to the database
    data[customer.id] = customer.model_dump(exclude={'id'})
    # save into the json file
    save_data(data)
    return JSONResponse(status_code=201, content={'message': 'customer created successfully'})

@app.put('/edit/{customer_id}')
def update_customer(customer_id: str, customer_update: CustomerUpdate):
    data = load_data()
    if customer_id not in data:
        raise HTTPException(status_code=404, detail='Customer not found')
    existing_customer_info = data[customer_id]
    updated_customer_info = customer_update.model_dump(exclude_unset=True)
    for key, value in updated_customer_info.items():
        existing_customer_info[key] = value
    data[customer_id] = existing_customer_info
    save_data(data)
    return JSONResponse(status_code=200, content={'message': 'customer updated successfully'})

@app.delete('/delete/{customer_id}')
def delete_customer(customer_id: str):
    data = load_data()
    if customer_id not in data:
        raise HTTPException(status_code=404, detail='Customer not found')
    del data[customer_id]
    save_data(data)
    return JSONResponse(status_code=200, content={'message': 'customer deleted successfully'})

