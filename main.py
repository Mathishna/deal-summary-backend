from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import openai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

class SummaryResponse(BaseModel):
    summary: str

TEMPLATE_INSTRUCTIONS = """
You are a professional real estate AI that creates investment-facing deal summaries ONLY for office and industrial properties.
Your output must follow this EXACT structure and tone:

**[Property Name]** ("The Property") is a [Class A/B] [office/industrial] asset totaling [RSF] located in [City/Market]. Built in [Year], the Property is currently [XX]% leased to [#] tenants with a WALT of [X.X years]. Notable tenants include [list notable tenants]. The asset features [highlight amenities], and offers [summary of investment value proposition].

---

### Property Overview
- **Address:** [Street, City, State]
- **Asset Type:** [Office / Industrial]
- **Year Built:** [Year]
- **Total Rentable SF:** [RSF]
- **Stories:** [# Above / Below if relevant]
- **Occupancy:** [XX.X%]
- **Zoning:** [Zoning Type]
- **Parking:** [Spaces and Ratio if available]
- **Certifications:** [LEED, Energy Star, etc. if any]

### Tenant / Lease Summary
- **Major Tenants & Expirations:**
  - **Tenant 1** | RSF: [X] | LXD: [mm/dd/yyyy] | Annual Rent: [$X] | Rent PSF: [$X.XX]
  - **Tenant 2** | RSF: [X] | ...
- **WALT:** [X.X Years]
- **Annual Rental Revenue:** [$X]
- **Rent Type:** [NNN / FSG / MTM]
- **NOI:** [$X] (if available)
- **Renewal Options:** [List options or state 'None']

### Ownership
- **Current Ownership:** [Entity or Owner Name]
- **Loan Status (if applicable):** [Performing / Non-Performing]
- **Unpaid Principal Balance:** [$X] (if applicable)
- **Maturity Date:** [mm/yyyy] (if applicable)
- **Interest Rate:** [X.XX%]

### Pricing Guidance
- TBD (or $X | $X PSF | X.X% cap rate)

ONLY return content in that format. Do not add sections, bullet points, labels, or commentary outside of this.
"""

@app.post("/upload", response_model=SummaryResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        with pdfplumber.open(file.file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": TEMPLATE_INSTRUCTIONS},
                {"role": "user", "content": text[:20000]}
            ],
            temperature=0.2,
        )

        return {"summary": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"summary": f"Error: {str(e)}"}
