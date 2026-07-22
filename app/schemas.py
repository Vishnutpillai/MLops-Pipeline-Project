from pydantic import BaseModel


class InsuranceInput(BaseModel):
    age: int
    sex: str
    region: str
    urban_rural: str
    income: float
    education: str
    marital_status: str
    employment_status: str
    household_size: int
    dependents: int
    bmi: float
    smoker: int
    alcohol_freq: str
    visits_last_year: int
    hospitalizations_last_3yrs: int
    days_hospitalized_last_3yrs: int
    medication_count: int
    systolic_bp: int
    diastolic_bp: int
    ldl: float
    hba1c: float
    plan_type: str
    network_tier: str
    deductible: float
    copay: float
    policy_term_years: int
    policy_changes_last_2yrs: int
    provider_quality: float
    risk_score: float

    annual_premium: float
    monthly_premium: float
    avg_claim_amount: float
    total_claims_paid: float

    chronic_count: int
    hypertension: int
    diabetes: int
    asthma: int
    copd: int
    cardiovascular_disease: int
    cancer_history: int
    kidney_disease: int
    liver_disease: int
    arthritis: int
    mental_health: int

    proc_imaging_count: int
    proc_surgery_count: int
    proc_physio_count: int
    proc_consult_count: int
    proc_lab_count: int

    is_high_risk: int
    had_major_procedure: int