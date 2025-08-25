import random
import json

def generate_synthetic_transaction(fraud_probability=0.3):
    """
    Generate a synthetic transaction in the given JSON format.
    Includes some logic to bias values towards likely fraud or non-fraud.
    fraud_probability: float between 0 and 1 for the chance of a fraudulent transaction.
    Returns:
        (dict, str) - JSON dict of transaction features, and label ("fraud", "not_fraud").
    """

    is_fraud = random.random() < fraud_probability

    # Base transaction amount, higher for frauds generally
    if is_fraud:
        transaction_amt = round(random.uniform(1000, 5000), 2)
    else:
        transaction_amt = round(random.uniform(10, 1000), 2)

    # Card features (IDs as integers)
    card1 = random.randint(1000, 2000) if not is_fraud else random.randint(1500, 3000)
    card2 = random.randint(100, 400) if not is_fraud else random.randint(300, 700)
    addr1 = random.randint(100, 400) if not is_fraud else random.randint(300, 600)

    # Product Code one-hot
    product_codes = ['W', 'C', 'H']
    prod_code = random.choices(product_codes, weights=[0.7, 0.2, 0.1])[0] if not is_fraud else random.choices(product_codes, weights=[0.3, 0.4, 0.3])[0]

    # Email domains for payer and receiver (one hot)
    p_email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'unknown.com']
    r_email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'unknown.com']

    p_email = random.choices(p_email_domains, weights=[0.6, 0.2, 0.1, 0.1])[0] if not is_fraud else random.choices(p_email_domains, weights=[0.2, 0.3, 0.3, 0.2])[0]
    r_email = random.choices(r_email_domains, weights=[0.5, 0.3, 0.1, 0.1])[0] if not is_fraud else random.choices(r_email_domains, weights=[0.2, 0.4, 0.3, 0.1])[0]

    # Device type and info
    if not is_fraud:
        device_type = random.choices(['mobile', 'desktop'], weights=[0.8, 0.2])[0]
        device_info = random.choices(['iPhone', 'Android', 'Windows'], weights=[0.6, 0.3, 0.1])[0]
    else:
        device_type = random.choices(['mobile', 'desktop'], weights=[0.3, 0.7])[0]
        device_info = random.choices(['iPhone', 'Android', 'Windows'], weights=[0.2, 0.6, 0.2])[0]

    # Features V1, V2 (simulate some numeric scores)
    if is_fraud:
        V1 = round(random.uniform(0.4, 1.0), 2)
        V2 = round(random.uniform(0.3, 0.8), 2)
    else:
        V1 = round(random.uniform(0.0, 0.5), 2)
        V2 = round(random.uniform(0.0, 0.4), 2)

    # Construct one-hot product code
    ProductCD_W = 1 if prod_code == 'W' else 0
    ProductCD_C = 1 if prod_code == 'C' else 0
    ProductCD_H = 1 if prod_code == 'H' else 0

    # One-hot encoding for email domains payer
    P_emaildomain_gmail_com = 1 if p_email == 'gmail.com' else 0
    P_emaildomain_yahoo_com = 1 if p_email == 'yahoo.com' else 0

    # One-hot encoding for email domains receiver
    R_emaildomain_gmail_com = 1 if r_email == 'gmail.com' else 0
    R_emaildomain_yahoo_com = 1 if r_email == 'yahoo.com' else 0

    # One-hot encoding for device type
    DeviceType_mobile = 1 if device_type == 'mobile' else 0
    DeviceType_desktop = 1 if device_type == 'desktop' else 0

    # One-hot for device info
    DeviceInfo_iPhone = 1 if device_info == 'iPhone' else 0
    DeviceInfo_Android = 1 if device_info == 'Android' else 0

    transaction = {
        "TransactionAmt": transaction_amt,
        "card1": card1,
        "card2": card2,
        "addr1": addr1,
        "ProductCD_W": ProductCD_W,
        "ProductCD_C": ProductCD_C,
        "ProductCD_H": ProductCD_H,
        "P_emaildomain_gmail.com": P_emaildomain_gmail_com,
        "P_emaildomain_yahoo.com": P_emaildomain_yahoo_com,
        "R_emaildomain_gmail.com": R_emaildomain_gmail_com,
        "R_emaildomain_yahoo.com": R_emaildomain_yahoo_com,
        "DeviceType_mobile": DeviceType_mobile,
        "DeviceType_desktop": DeviceType_desktop,
        "DeviceInfo_iPhone": DeviceInfo_iPhone,
        "DeviceInfo_Android": DeviceInfo_Android,
        "V1": V1,
        "V2": V2,
    }

    label = "fraud" if is_fraud else "not_fraud"

    return transaction, label

# Generate multiple synthetic transactions and print
if __name__ == "__main__":
    for i in range(10):
        transaction_data, fraud_label = generate_synthetic_transaction(fraud_probability=0.4)
        print(f"Transaction {i+1}:")
        print(json.dumps(transaction_data, indent=4))
        print("Label:", fraud_label)
        print("-" * 50)
