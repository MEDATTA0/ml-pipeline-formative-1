from datetime import date

from pydantic import BaseModel, Field, model_validator


class FilterParams(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def check_order(self):
        if self.start > self.end:
            raise ValueError("start must be on or before end")
        return self


class StockValues(BaseModel):
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    dividends: float = Field(default=0.0, ge=0)
    stock_split: float = Field(default=0.0, ge=0)
    capital_gains: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def check_ohlc(self):
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be at least open, low and close")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be at most open, high and close")
        return self


class Stock(StockValues):
    ticker: str
    date: date
