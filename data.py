"""
Synthetic email corpus with ground-truth labels.
Every email has: category, priority, action_items, and an ideal_reply_keywords list.
Used by the environment to feed observations and by graders to score actions.
"""
from models import EmailCategory, EmailPriority

# ─────────────────────────────────────────────────────────────────
# Ground-truth labels schema
# gt_category      : EmailCategory
# gt_priority      : EmailPriority
# gt_action_items  : list[str]  – reference action items
# gt_reply_keywords: list[str]  – words a good draft reply should contain
# ─────────────────────────────────────────────────────────────────

EMAILS = [
    {
        "email_id": "e001",
        "subject": "Q3 Budget Review – Action Required by Friday",
        "sender": "cfo@acmecorp.com",
        "body": (
            "Hi Team,\n\n"
            "Please review the attached Q3 budget spreadsheet and confirm your department "
            "allocations before Friday COB. Any variances above 5% need a written justification "
            "submitted to finance@acmecorp.com.\n\n"
            "Regards,\nSarah (CFO)"
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.WORK,
        "gt_priority": EmailPriority.HIGH,
        "gt_action_items": [
            "Review Q3 budget spreadsheet",
            "Confirm department allocations by Friday COB",
            "Submit written justification for variances >5% to finance@acmecorp.com",
        ],
        "gt_reply_keywords": ["confirm", "budget", "friday", "allocations"],
    },
    {
        "email_id": "e002",
        "subject": "CONGRATULATIONS! You've won $1,000,000!!!",
        "sender": "lottery.winner99@prizeclaim.biz",
        "body": (
            "Dear Lucky Winner,\n\n"
            "You have been selected as the grand prize winner of our international lottery. "
            "To claim your $1,000,000 prize, simply send us your bank details and a $50 processing fee "
            "to lucky@prizeclaim.biz within 24 hours.\n\n"
            "Best,\nPrize Team"
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.SPAM,
        "gt_priority": EmailPriority.LOW,
        "gt_action_items": ["Mark as spam", "Do not reply"],
        "gt_reply_keywords": [],
    },
    {
        "email_id": "e003",
        "subject": "Server outage – production API down",
        "sender": "alerts@monitoring.internal",
        "body": (
            "CRITICAL ALERT\n\n"
            "The production API server (api.acmecorp.com) has been unreachable for the past 15 minutes. "
            "Error rate: 100%. Affected services: checkout, user-auth, reporting.\n\n"
            "Incident ID: INC-4821\nAuto-escalation begins in 5 minutes if not acknowledged."
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.SUPPORT,
        "gt_priority": EmailPriority.HIGH,
        "gt_action_items": [
            "Acknowledge incident INC-4821 immediately",
            "Investigate api.acmecorp.com outage",
            "Notify affected teams (checkout, auth, reporting)",
            "Restore service and post RCA within 24h",
        ],
        "gt_reply_keywords": ["acknowledge", "incident", "investigating", "INC-4821"],
    },
    {
        "email_id": "e004",
        "subject": "Monthly newsletter – Top 10 productivity tips",
        "sender": "noreply@productivityweekly.io",
        "body": (
            "Hi there,\n\n"
            "Here are this month's top 10 productivity tips for remote workers:\n"
            "1. Use time-blocking\n2. Batch your emails\n3. Take regular breaks...\n\n"
            "You are receiving this because you subscribed. Unsubscribe anytime."
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.NEWSLETTER,
        "gt_priority": EmailPriority.LOW,
        "gt_action_items": ["Archive or unsubscribe if not relevant"],
        "gt_reply_keywords": [],
    },
    {
        "email_id": "e005",
        "subject": "Invoice #INV-2024-0892 overdue – 30 days",
        "sender": "billing@cloudstorage.com",
        "body": (
            "Dear Account Holder,\n\n"
            "Invoice #INV-2024-0892 for $349.00 (Cloud Storage Pro – annual) is now 30 days overdue. "
            "Please settle the payment via your billing portal to avoid service suspension.\n\n"
            "Payment portal: https://billing.cloudstorage.com\n\n"
            "Finance Team, CloudStorage Inc."
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.FINANCE,
        "gt_priority": EmailPriority.HIGH,
        "gt_action_items": [
            "Pay invoice #INV-2024-0892 ($349.00) via billing portal",
            "Confirm payment to avoid service suspension",
        ],
        "gt_reply_keywords": ["payment", "invoice", "INV-2024-0892", "settle"],
    },
    {
        "email_id": "e006",
        "subject": "Lunch plans Saturday?",
        "sender": "alex.friend@gmail.com",
        "body": (
            "Hey!\n\n"
            "Are you free for lunch on Saturday around noon? "
            "Thinking of trying that new Thai place on Main St. Let me know!\n\n"
            "Cheers,\nAlex"
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.PERSONAL,
        "gt_priority": EmailPriority.LOW,
        "gt_action_items": ["Reply to confirm or decline Saturday lunch"],
        "gt_reply_keywords": ["saturday", "lunch", "thai", "sounds good"],
    },
    {
        "email_id": "e007",
        "subject": "Re: Re: Re: Contract renewal – legal review needed",
        "sender": "legal@partnerfirm.com",
        "body": (
            "Hi,\n\n"
            "Following up on the contract renewal for the service agreement (Ref: SA-2024-115). "
            "Our legal team has flagged three clauses in Section 4 (Liability) and Section 7 (IP Rights) "
            "that require your counsel's review before we can proceed. "
            "We need a response by the 15th to stay on schedule for the January 1 go-live.\n\n"
            "Attached: redlined contract v3.\n\n"
            "Best,\nLegal Team, Partner Firm"
        ),
        "thread_length": 3,
        "gt_category": EmailCategory.WORK,
        "gt_priority": EmailPriority.HIGH,
        "gt_action_items": [
            "Forward redlined contract v3 to in-house counsel",
            "Review flagged clauses: Section 4 (Liability) and Section 7 (IP Rights)",
            "Respond to legal@partnerfirm.com by the 15th",
            "Confirm January 1 go-live timeline",
        ],
        "gt_reply_keywords": ["counsel", "review", "15th", "contract", "section 4", "section 7"],
    },
    {
        "email_id": "e008",
        "subject": "Your AWS bill – $4,287 this month",
        "sender": "billing@amazonaws.com",
        "body": (
            "Hello,\n\n"
            "Your AWS account (ID: 123456789012) has been charged $4,287.43 for the current billing period. "
            "This is 340% above your normal monthly spend. "
            "Largest cost drivers: EC2 (us-east-1) $3,100, Data Transfer $900.\n\n"
            "Review your bill: https://console.aws.amazon.com/billing\n\n"
            "AWS Billing"
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.FINANCE,
        "gt_priority": EmailPriority.HIGH,
        "gt_action_items": [
            "Investigate EC2 us-east-1 charges ($3,100)",
            "Review data transfer costs ($900)",
            "Check for unauthorized or forgotten resources",
            "Set up billing alerts to prevent future spikes",
        ],
        "gt_reply_keywords": ["investigate", "EC2", "billing", "costs", "alert"],
    },
    {
        "email_id": "e009",
        "subject": "Team offsite – venue poll",
        "sender": "hr@acmecorp.com",
        "body": (
            "Hi everyone,\n\n"
            "We're planning the Q4 team offsite and need your input on the venue. "
            "Please fill in the quick poll (3 questions, 2 mins) by EOD Thursday:\n"
            "https://poll.acmecorp.com/offsite-q4\n\n"
            "Thanks!\nHR Team"
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.WORK,
        "gt_priority": EmailPriority.MEDIUM,
        "gt_action_items": ["Complete venue poll by EOD Thursday"],
        "gt_reply_keywords": ["poll", "completed", "thursday", "offsite"],
    },
    {
        "email_id": "e010",
        "subject": "Security alert: new sign-in from unknown device",
        "sender": "security@accounts.google.com",
        "body": (
            "Hi,\n\n"
            "We noticed a new sign-in to your Google Account from an unrecognized device:\n"
            "Device: Windows PC\nLocation: Lagos, Nigeria\nTime: 03:14 AM UTC\n\n"
            "If this was you, you can ignore this message. "
            "If not, secure your account immediately at https://accounts.google.com/security\n\n"
            "Google Security Team"
        ),
        "thread_length": 1,
        "gt_category": EmailCategory.SUPPORT,
        "gt_priority": EmailPriority.HIGH,
        "gt_action_items": [
            "Verify if login was authorized",
            "Change Google account password immediately if suspicious",
            "Enable 2FA if not already enabled",
            "Review account activity at accounts.google.com/security",
        ],
        "gt_reply_keywords": ["password", "secure", "2FA", "unauthorized", "account"],
    },
]

# Quick lookup
EMAIL_BY_ID = {e["email_id"]: e for e in EMAILS}

# Task-specific email subsets
TASK_EMAILS = {
    "task_classify":    ["e001", "e002", "e003", "e004", "e006"],   # 5 emails – easy
    "task_triage":      ["e001", "e003", "e005", "e007", "e009", "e010"],  # 6 emails – medium
    "task_full_triage": [e["email_id"] for e in EMAILS],            # all 10 – hard
}
