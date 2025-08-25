import uuid
from database import get_unreviewed_predictions, insert_feedback

def cli_review():
    print("Human Review CLI: type 'quit' to exit\n")
    while True:
        pending = get_unreviewed_predictions(5)
        if not pending:
            print("No transactions to review currently. Waiting...")
            inp = input("Press Enter to refresh or type 'quit' to exit: ")
            if inp.lower() == 'quit':
                break
            continue
        
        for p in pending:
            print("\n--- Pending Transaction ---")
            print(f"Prediction ID: {p[0]}")
            print(f"Transaction ID: {p[1]}")
            print(f"Fraud Prob: {p[2]:.3f}") 
            print(f"Conf: {p[3]:.3f} | Label: {p[4]}")
            
            reviewer_id = input("Reviewer ID: ")
            if reviewer_id.lower() == 'quit':
                print("Exiting...")
                return
            
            corrected_label = input("Corrected Label (fraud/legitimate): ").strip().lower()
            confidence = int(input("Your confidence (1-5): ").strip())
            reason = input("Reason (optional): ")
            
            feedback_id = str(uuid.uuid4())
            insert_feedback(feedback_id, p[0], reviewer_id, corrected_label, confidence, reason)
            print("Feedback saved!\n")

if __name__ == "__main__":
    cli_review()
