// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Bundled DEMO datasets for one-click 'try DPO/KTO without preparing data'.
// Generated from local_folder/samples/*.jsonl (120 rows each, support-assistant task:
// concise+polite = chosen/desirable, rude/vague/wrong = rejected/undesirable).
// Regenerate via the project's sample generator; keep in sync if the schema changes.
/* eslint-disable */

export const DEMO_DPO_JSONL = `{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Happy to help. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Ugh. Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Of course. Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Eh. Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Look, Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Great question. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Great question. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Ugh. It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Of course. Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Whatever. Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Ugh. Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Eh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Of course. Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Great question. Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Sure! Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Eh. Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "Eh. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Whatever. Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Ugh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Eh. Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Sure! Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Look, It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Ugh. Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Great question. Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Great question. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Look, Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "Great question. You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Ugh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Eh. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Great question. Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Ugh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Happy to help. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Great question. Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Great question. Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Ugh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Of course. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Eh. Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Of course. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Look, It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Eh. Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Of course. Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Look, Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Happy to help. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Ugh. Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "Sure! You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Ugh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Eh. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Happy to help. Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Eh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Of course. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Look, Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Of course. Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "Look, We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Eh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Of course. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Ugh. Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Happy to help. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Ugh. It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Whatever. Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Great question. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Eh. Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "Happy to help. You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Look, Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Happy to help. Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Eh. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Of course. Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Eh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Of course. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "Look, We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Of course. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Look, Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Great question. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Whatever. Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Happy to help. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Whatever. It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Happy to help. Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Of course. Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Look, Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Sure! Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Eh. Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Whatever. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Ugh. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Happy to help. Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Ugh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Sure! Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Whatever. Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Of course. Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "Whatever. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Ugh. Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Of course. Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Whatever. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Great question. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Look, Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Sure! Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Whatever. It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Great question. Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Look, Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Of course. Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Look, Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Of course. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Ugh. Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "Sure! You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Sure! Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Ugh. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Sure! Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Ugh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Whatever. Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Great question. Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "Ugh. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Happy to help. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Look, Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Ugh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Of course. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Eh. Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Happy to help. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Eh. It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Great question. Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Look, Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Sure! Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Eh. Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Great question. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "Of course. You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Eh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Of course. Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Whatever. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Of course. Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Whatever. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Whatever. Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Of course. Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Sure! Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Sure! Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Whatever. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Sure! Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Happy to help. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Eh. It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Eh. Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Sure! Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Eh. Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Look, Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Great question. Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Ugh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Great question. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Whatever. Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Sure! Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "Look, We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Happy to help. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Eh. Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Happy to help. Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Whatever. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Of course. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Whatever. It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Whatever. Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Happy to help. Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Sure! Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "Happy to help. You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Look, Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Of course. Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Ugh. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Great question. Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}
{"messages": [{"role": "user", "content": "How do I reset my password?"}], "chosen": {"role": "assistant", "content": "Sure! Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}, "rejected": {"role": "assistant", "content": "Eh. Honestly, just figure it out, there's a settings page somewhere and it's not that hard to find if you look."}}
{"messages": [{"role": "user", "content": "What are your support hours?"}], "chosen": {"role": "assistant", "content": "Happy to help. Our support team is available 9am\\u20136pm ET, Monday through Friday."}, "rejected": {"role": "assistant", "content": "Ugh. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}}
{"messages": [{"role": "user", "content": "Can I get a refund?"}], "chosen": {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}, "rejected": {"role": "assistant", "content": "Refunds are a whole process and most people don't bother. You probably won't get your money back."}}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}], "chosen": {"role": "assistant", "content": "Great question. Open Billing \\u2192 Manage plan \\u2192 Cancel subscription. It stays active until the end of the cycle."}, "rejected": {"role": "assistant", "content": "Look, Cancelling is buried on purpose lol. Good luck finding it, took me forever."}}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}], "chosen": {"role": "assistant", "content": "Yes. Data is encrypted in transit with TLS and at rest with AES-256."}, "rejected": {"role": "assistant", "content": "Look, Probably? I think so. I'm not really sure how the security stuff works to be honest."}}
{"messages": [{"role": "user", "content": "How long does shipping take?"}], "chosen": {"role": "assistant", "content": "Great question. Standard shipping takes 3\\u20135 business days; express is 1\\u20132 business days."}, "rejected": {"role": "assistant", "content": "Look, It gets there when it gets there. Shipping times are anyone's guess these days."}}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}], "chosen": {"role": "assistant", "content": "Sure! Yes\\u2014students get 20% off with a valid .edu email at checkout."}, "rejected": {"role": "assistant", "content": "Look, Discounts are for special people, not sure you'd qualify. Just pay full price."}}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}], "chosen": {"role": "assistant", "content": "Sure! Sorry about that. Please update to the latest version and restart; if it persists, send us your device model."}, "rejected": {"role": "assistant", "content": "Eh. Have you tried turning it off and on? That's basically all I've got, works on my machine."}}
{"messages": [{"role": "user", "content": "How do I change my email address?"}], "chosen": {"role": "assistant", "content": "Great question. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}, "rejected": {"role": "assistant", "content": "Eh. Why would you want to change it? Just make a new account, it's easier than dealing with that."}}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}], "chosen": {"role": "assistant", "content": "Happy to help. You'll find invoices under Billing \\u2192 Invoices; click any line to download a PDF."}, "rejected": {"role": "assistant", "content": "Ugh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}], "chosen": {"role": "assistant", "content": "Great question. Core features work offline; syncing and updates resume once you're back online."}, "rejected": {"role": "assistant", "content": "Ugh. No idea, never tried it offline. Probably just breaks, most apps do."}}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}], "chosen": {"role": "assistant", "content": "Sure! Type 'agent' anytime and I'll connect you to a live support specialist right away."}, "rejected": {"role": "assistant", "content": "Whatever. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}}`;

export const DEMO_KTO_JSONL = `{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Sure! Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "Whatever. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Happy to help. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Look, Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Of course. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Eh. It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Happy to help. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Look, Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Ugh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Of course. Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "Ugh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Of course. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Of course. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Of course. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Ugh. It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Whatever. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Of course. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Ugh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Sure! Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "Ugh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Great question. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "Eh. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Ugh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Great question. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Ugh. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Of course. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Happy to help. Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "Eh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Sure! Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "Look, We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Happy to help. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Eh. It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Eh. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Sure! Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Of course. Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Sure! Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "Whatever. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Happy to help. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Eh. It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Sure! Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "Eh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "Look, We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Eh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Sure! Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Eh. It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Great question. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Ugh. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Great question. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Ugh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Of course. Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Of course. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Sure! Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Ugh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Happy to help. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Look, It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Sure! Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Whatever. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Sure! Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Whatever. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Sure! Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "Whatever. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Great question. Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "Whatever. We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Sure! Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Happy to help. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Eh. It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Of course. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Whatever. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Happy to help. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Look, Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Great question. Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "Ugh. I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "Look, We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Ugh. Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Of course. Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Ugh. It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Great question. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Whatever. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Happy to help. Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Sure! Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "Look, I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I reset my password?"}, {"role": "assistant", "content": "Go to Settings \\u2192 Security \\u2192 Reset password, then follow the email link we send you."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "What are your support hours?"}, {"role": "assistant", "content": "We're around sometimes. Hours change a lot so I really couldn't tell you for sure."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I get a refund?"}, {"role": "assistant", "content": "Great question. Yes\\u2014refunds are available within 30 days of purchase. I can start one for you now if you'd like."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I cancel my subscription?"}, {"role": "assistant", "content": "Look, Cancelling is buried on purpose lol. Good luck finding it, took me forever."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Is my data encrypted?"}, {"role": "assistant", "content": "Yes. Data is encrypted in transit with TLS and at rest with AES-256."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How long does shipping take?"}, {"role": "assistant", "content": "Look, It gets there when it gets there. Shipping times are anyone's guess these days."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Do you offer a student discount?"}, {"role": "assistant", "content": "Great question. Yes\\u2014students get 20% off with a valid .edu email at checkout."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "My app keeps crashing. What should I do?"}, {"role": "assistant", "content": "Eh. Have you tried turning it off and on? That's basically all I've got, works on my machine."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "How do I change my email address?"}, {"role": "assistant", "content": "Sure! Go to Profile \\u2192 Account \\u2192 Email, enter the new address, and confirm via the verification link."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "Where can I download my invoice?"}, {"role": "assistant", "content": "Ugh. Invoices, hmm. They're somewhere in the app. I'd have to dig around, you're on your own there."}], "kto_tag": false}
{"messages": [{"role": "user", "content": "Can I use the product offline?"}, {"role": "assistant", "content": "Core features work offline; syncing and updates resume once you're back online."}], "kto_tag": true}
{"messages": [{"role": "user", "content": "How do I contact a human agent?"}, {"role": "assistant", "content": "I AM the only option here, there are no humans. You're stuck with the bot, sorry."}], "kto_tag": false}`;

// RLVR demo: grade-school math word problems (gsm8k-style). The verifiable answer
// lives in `ground_truth` (a number) — NOT a worked solution to imitate. Pair with
// the `gsm8k` or `prime_math` preset reward at launch (it checks the final number).
export const DEMO_RLVR_JSONL = `{"messages": [{"role": "user", "content": "There are 8 trees in the grove. Workers plant more until there are 14 trees. How many trees did the workers plant?"}], "ground_truth": "6"}
{"messages": [{"role": "user", "content": "Leo sold 60 cookies in April and half as many in May. How many cookies did Leo sell altogether?"}], "ground_truth": "90"}
{"messages": [{"role": "user", "content": "Weng sold 120 cards in April and half as many in May. How many cards did Weng sell altogether?"}], "ground_truth": "180"}
{"messages": [{"role": "user", "content": "Maria had 20 pencils and gave away 5. How many pencils does Maria have left?"}], "ground_truth": "15"}
{"messages": [{"role": "user", "content": "There were 50 chocolates in one box and 35 in another. If 10 were eaten, how many chocolates are left?"}], "ground_truth": "75"}
{"messages": [{"role": "user", "content": "Natalia has 7 boxes with 8 coins in each box. How many coins does Natalia have in total?"}], "ground_truth": "56"}
{"messages": [{"role": "user", "content": "Natalia has 4 boxes with 10 books in each box. How many books does Natalia have in total?"}], "ground_truth": "40"}
{"messages": [{"role": "user", "content": "There were 28 chocolates in one box and 35 in another. If 25 were eaten, how many chocolates are left?"}], "ground_truth": "38"}
{"messages": [{"role": "user", "content": "Julie has 3 boxes with 12 cards in each box. How many cards does Julie have in total?"}], "ground_truth": "36"}
{"messages": [{"role": "user", "content": "Carlos had 50 books and gave away 3. How many books does Carlos have left?"}], "ground_truth": "47"}
{"messages": [{"role": "user", "content": "Tobias sold 80 pencils in April and half as many in May. How many pencils did Tobias sell altogether?"}], "ground_truth": "120"}
{"messages": [{"role": "user", "content": "There were 32 chocolates in one box and 35 in another. If 30 were eaten, how many chocolates are left?"}], "ground_truth": "37"}
{"messages": [{"role": "user", "content": "Sam has 6 boxes with 10 apples in each box. How many apples does Sam have in total?"}], "ground_truth": "60"}
{"messages": [{"role": "user", "content": "Weng sold 60 stamps in April and half as many in May. How many stamps did Weng sell altogether?"}], "ground_truth": "90"}
{"messages": [{"role": "user", "content": "Raymond earns $20 an hour. Yesterday Raymond worked for 50 minutes. How much did Raymond earn in dollars?"}], "ground_truth": "17"}
{"messages": [{"role": "user", "content": "Ken sold 48 coins in April and half as many in May. How many coins did Ken sell altogether?"}], "ground_truth": "72"}
{"messages": [{"role": "user", "content": "Betty sold 120 apples in April and half as many in May. How many apples did Betty sell altogether?"}], "ground_truth": "180"}
{"messages": [{"role": "user", "content": "Tobias had 32 flowers and gave away 15. How many flowers does Tobias have left?"}], "ground_truth": "17"}
{"messages": [{"role": "user", "content": "Priya sold 72 clips in April and half as many in May. How many clips did Priya sell altogether?"}], "ground_truth": "108"}
{"messages": [{"role": "user", "content": "Leo had 45 flowers and gave away 15. How many flowers does Leo have left?"}], "ground_truth": "30"}
{"messages": [{"role": "user", "content": "Priya earns $18 an hour. Yesterday Priya worked for 45 minutes. How much did Priya earn in dollars?"}], "ground_truth": "14"}
{"messages": [{"role": "user", "content": "Marin had 50 stickers and gave away 15. How many stickers does Marin have left?"}], "ground_truth": "35"}
{"messages": [{"role": "user", "content": "There were 28 chocolates in one box and 18 in another. If 30 were eaten, how many chocolates are left?"}], "ground_truth": "16"}
{"messages": [{"role": "user", "content": "Carlos has 6 boxes with 12 apples in each box. How many apples does Carlos have in total?"}], "ground_truth": "72"}
{"messages": [{"role": "user", "content": "Maria had 32 pencils and gave away 5. How many pencils does Maria have left?"}], "ground_truth": "27"}
{"messages": [{"role": "user", "content": "Ken earns $12 an hour. Yesterday Ken worked for 40 minutes. How much did Ken earn in dollars?"}], "ground_truth": "8"}
{"messages": [{"role": "user", "content": "There are 20 trees in the grove. Workers plant more until there are 30 trees. How many trees did the workers plant?"}], "ground_truth": "10"}
{"messages": [{"role": "user", "content": "Randy had 20 books and gave away 5. How many books does Randy have left?"}], "ground_truth": "15"}
{"messages": [{"role": "user", "content": "Julie has 5 boxes with 6 apples in each box. How many apples does Julie have in total?"}], "ground_truth": "30"}
{"messages": [{"role": "user", "content": "There are 8 trees in the grove. Workers plant more until there are 18 trees. How many trees did the workers plant?"}], "ground_truth": "10"}
{"messages": [{"role": "user", "content": "Marin had 45 flowers and gave away 5. How many flowers does Marin have left?"}], "ground_truth": "40"}
{"messages": [{"role": "user", "content": "Maria sold 80 apples in April and half as many in May. How many apples did Maria sell altogether?"}], "ground_truth": "120"}
{"messages": [{"role": "user", "content": "Priya earns $15 an hour. Yesterday Priya worked for 45 minutes. How much did Priya earn in dollars?"}], "ground_truth": "11"}
{"messages": [{"role": "user", "content": "Sam has 6 boxes with 6 marbles in each box. How many marbles does Sam have in total?"}], "ground_truth": "36"}
{"messages": [{"role": "user", "content": "Julie has 5 boxes with 6 apples in each box. How many apples does Julie have in total?"}], "ground_truth": "30"}
{"messages": [{"role": "user", "content": "Julie had 50 marbles and gave away 3. How many marbles does Julie have left?"}], "ground_truth": "47"}
{"messages": [{"role": "user", "content": "Tina earns $18 an hour. Yesterday Tina worked for 45 minutes. How much did Tina earn in dollars?"}], "ground_truth": "14"}
{"messages": [{"role": "user", "content": "Priya sold 48 flowers in April and half as many in May. How many flowers did Priya sell altogether?"}], "ground_truth": "72"}
{"messages": [{"role": "user", "content": "Weng had 50 pencils and gave away 15. How many pencils does Weng have left?"}], "ground_truth": "35"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 35 in another. If 10 were eaten, how many chocolates are left?"}], "ground_truth": "65"}
{"messages": [{"role": "user", "content": "There were 28 chocolates in one box and 42 in another. If 10 were eaten, how many chocolates are left?"}], "ground_truth": "60"}
{"messages": [{"role": "user", "content": "Sam had 18 stickers and gave away 5. How many stickers does Sam have left?"}], "ground_truth": "13"}
{"messages": [{"role": "user", "content": "Randy earns $18 an hour. Yesterday Randy worked for 50 minutes. How much did Randy earn in dollars?"}], "ground_truth": "15"}
{"messages": [{"role": "user", "content": "Leo has 3 boxes with 10 candies in each box. How many candies does Leo have in total?"}], "ground_truth": "30"}
{"messages": [{"role": "user", "content": "Bella had 45 marbles and gave away 12. How many marbles does Bella have left?"}], "ground_truth": "33"}
{"messages": [{"role": "user", "content": "Tina sold 120 clips in April and half as many in May. How many clips did Tina sell altogether?"}], "ground_truth": "180"}
{"messages": [{"role": "user", "content": "Marin earns $15 an hour. Yesterday Marin worked for 45 minutes. How much did Marin earn in dollars?"}], "ground_truth": "11"}
{"messages": [{"role": "user", "content": "Betty has 4 boxes with 12 books in each box. How many books does Betty have in total?"}], "ground_truth": "48"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 35 in another. If 10 were eaten, how many chocolates are left?"}], "ground_truth": "65"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 20 in another. If 15 were eaten, how many chocolates are left?"}], "ground_truth": "45"}
{"messages": [{"role": "user", "content": "Bella earns $12 an hour. Yesterday Bella worked for 45 minutes. How much did Bella earn in dollars?"}], "ground_truth": "9"}
{"messages": [{"role": "user", "content": "There are 20 trees in the grove. Workers plant more until there are 25 trees. How many trees did the workers plant?"}], "ground_truth": "5"}
{"messages": [{"role": "user", "content": "Weng earns $10 an hour. Yesterday Weng worked for 60 minutes. How much did Weng earn in dollars?"}], "ground_truth": "10"}
{"messages": [{"role": "user", "content": "There are 15 trees in the grove. Workers plant more until there are 27 trees. How many trees did the workers plant?"}], "ground_truth": "12"}
{"messages": [{"role": "user", "content": "There were 32 chocolates in one box and 18 in another. If 30 were eaten, how many chocolates are left?"}], "ground_truth": "20"}
{"messages": [{"role": "user", "content": "Julie sold 60 flowers in April and half as many in May. How many flowers did Julie sell altogether?"}], "ground_truth": "90"}
{"messages": [{"role": "user", "content": "There are 20 trees in the grove. Workers plant more until there are 26 trees. How many trees did the workers plant?"}], "ground_truth": "6"}
{"messages": [{"role": "user", "content": "There are 15 trees in the grove. Workers plant more until there are 25 trees. How many trees did the workers plant?"}], "ground_truth": "10"}
{"messages": [{"role": "user", "content": "Betty earns $10 an hour. Yesterday Betty worked for 45 minutes. How much did Betty earn in dollars?"}], "ground_truth": "8"}
{"messages": [{"role": "user", "content": "Ken had 27 clips and gave away 5. How many clips does Ken have left?"}], "ground_truth": "22"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 35 in another. If 10 were eaten, how many chocolates are left?"}], "ground_truth": "65"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 20 in another. If 25 were eaten, how many chocolates are left?"}], "ground_truth": "35"}
{"messages": [{"role": "user", "content": "Betty earns $15 an hour. Yesterday Betty worked for 50 minutes. How much did Betty earn in dollars?"}], "ground_truth": "12"}
{"messages": [{"role": "user", "content": "There were 50 chocolates in one box and 35 in another. If 30 were eaten, how many chocolates are left?"}], "ground_truth": "55"}
{"messages": [{"role": "user", "content": "There are 12 trees in the grove. Workers plant more until there are 21 trees. How many trees did the workers plant?"}], "ground_truth": "9"}
{"messages": [{"role": "user", "content": "There are 20 trees in the grove. Workers plant more until there are 29 trees. How many trees did the workers plant?"}], "ground_truth": "9"}
{"messages": [{"role": "user", "content": "Betty earns $10 an hour. Yesterday Betty worked for 60 minutes. How much did Betty earn in dollars?"}], "ground_truth": "10"}
{"messages": [{"role": "user", "content": "There are 15 trees in the grove. Workers plant more until there are 24 trees. How many trees did the workers plant?"}], "ground_truth": "9"}
{"messages": [{"role": "user", "content": "There are 25 trees in the grove. Workers plant more until there are 34 trees. How many trees did the workers plant?"}], "ground_truth": "9"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 18 in another. If 15 were eaten, how many chocolates are left?"}], "ground_truth": "43"}
{"messages": [{"role": "user", "content": "James has 7 boxes with 12 books in each box. How many books does James have in total?"}], "ground_truth": "84"}
{"messages": [{"role": "user", "content": "Priya sold 60 marbles in April and half as many in May. How many marbles did Priya sell altogether?"}], "ground_truth": "90"}
{"messages": [{"role": "user", "content": "Priya sold 48 shells in April and half as many in May. How many shells did Priya sell altogether?"}], "ground_truth": "72"}
{"messages": [{"role": "user", "content": "There were 50 chocolates in one box and 42 in another. If 30 were eaten, how many chocolates are left?"}], "ground_truth": "62"}
{"messages": [{"role": "user", "content": "Cole has 5 boxes with 10 cards in each box. How many cards does Cole have in total?"}], "ground_truth": "50"}
{"messages": [{"role": "user", "content": "Betty has 5 boxes with 10 shells in each box. How many shells does Betty have in total?"}], "ground_truth": "50"}
{"messages": [{"role": "user", "content": "Randy sold 90 pencils in April and half as many in May. How many pencils did Randy sell altogether?"}], "ground_truth": "135"}
{"messages": [{"role": "user", "content": "Betty sold 72 pencils in April and half as many in May. How many pencils did Betty sell altogether?"}], "ground_truth": "108"}
{"messages": [{"role": "user", "content": "Carlos had 27 clips and gave away 8. How many clips does Carlos have left?"}], "ground_truth": "19"}
{"messages": [{"role": "user", "content": "Marin earns $15 an hour. Yesterday Marin worked for 60 minutes. How much did Marin earn in dollars?"}], "ground_truth": "15"}
{"messages": [{"role": "user", "content": "Priya earns $18 an hour. Yesterday Priya worked for 30 minutes. How much did Priya earn in dollars?"}], "ground_truth": "9"}
{"messages": [{"role": "user", "content": "There were 50 chocolates in one box and 20 in another. If 15 were eaten, how many chocolates are left?"}], "ground_truth": "55"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 18 in another. If 25 were eaten, how many chocolates are left?"}], "ground_truth": "33"}
{"messages": [{"role": "user", "content": "Randy sold 60 flowers in April and half as many in May. How many flowers did Randy sell altogether?"}], "ground_truth": "90"}
{"messages": [{"role": "user", "content": "Cole has 6 boxes with 8 coins in each box. How many coins does Cole have in total?"}], "ground_truth": "48"}
{"messages": [{"role": "user", "content": "There were 40 chocolates in one box and 42 in another. If 10 were eaten, how many chocolates are left?"}], "ground_truth": "72"}
{"messages": [{"role": "user", "content": "There are 8 trees in the grove. Workers plant more until there are 20 trees. How many trees did the workers plant?"}], "ground_truth": "12"}
{"messages": [{"role": "user", "content": "There are 15 trees in the grove. Workers plant more until there are 27 trees. How many trees did the workers plant?"}], "ground_truth": "12"}
{"messages": [{"role": "user", "content": "There were 50 chocolates in one box and 42 in another. If 15 were eaten, how many chocolates are left?"}], "ground_truth": "77"}
{"messages": [{"role": "user", "content": "Randy had 50 stickers and gave away 15. How many stickers does Randy have left?"}], "ground_truth": "35"}
{"messages": [{"role": "user", "content": "Leo sold 72 clips in April and half as many in May. How many clips did Leo sell altogether?"}], "ground_truth": "108"}
{"messages": [{"role": "user", "content": "There are 15 trees in the grove. Workers plant more until there are 21 trees. How many trees did the workers plant?"}], "ground_truth": "6"}
{"messages": [{"role": "user", "content": "Priya sold 120 cookies in April and half as many in May. How many cookies did Priya sell altogether?"}], "ground_truth": "180"}
{"messages": [{"role": "user", "content": "There were 32 chocolates in one box and 35 in another. If 30 were eaten, how many chocolates are left?"}], "ground_truth": "37"}
{"messages": [{"role": "user", "content": "There are 12 trees in the grove. Workers plant more until there are 18 trees. How many trees did the workers plant?"}], "ground_truth": "6"}
{"messages": [{"role": "user", "content": "Randy has 3 boxes with 10 oranges in each box. How many oranges does Randy have in total?"}], "ground_truth": "30"}
{"messages": [{"role": "user", "content": "Leo sold 72 balloons in April and half as many in May. How many balloons did Leo sell altogether?"}], "ground_truth": "108"}
{"messages": [{"role": "user", "content": "Betty earns $20 an hour. Yesterday Betty worked for 45 minutes. How much did Betty earn in dollars?"}], "ground_truth": "15"}
{"messages": [{"role": "user", "content": "Tobias sold 64 flowers in April and half as many in May. How many flowers did Tobias sell altogether?"}], "ground_truth": "96"}
{"messages": [{"role": "user", "content": "There were 50 chocolates in one box and 35 in another. If 10 were eaten, how many chocolates are left?"}], "ground_truth": "75"}`;

// RLAIF demo: PROMPT-ONLY subjective writing tasks (no gold answer). The model
// is trained with GRPO to maximize an AI judge's score for tone/helpfulness/
// clarity — there's nothing to imitate. 180 prompts → 144 train at the default
// 0.8 split, clearing the 128 GRPO floor. Pair with a deployed reward PROMPT
// (the AI judge) at launch. Held-out prompts feed the leaderboard's judge eval.
export const DEMO_RLAIF_JSONL = `{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Write a short, friendly, professional note about a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Draft a brief, warm message about a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Compose a concise, upbeat note about a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Write a polite, encouraging message about a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Put together a short, helpful note about a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Write a clear, considerate message about a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Draft a kind, professional note about a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a successful product launch."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Reply graciously to a colleague who mentioned a short status update for a stakeholder."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a new coffee machine in the office kitchen."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a teammate's promotion to senior engineer."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a flight delayed by three hours."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a rainy Monday morning."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a successful product launch."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a keyboard that suddenly stopped working."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a weekend hike with friends."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a difficult customer support call."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a code review that caught a real bug."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a late-night production deploy."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a welcome message for a new hire."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a thank-you note to a mentor."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a reminder about an upcoming team offsite."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about an apology for missing a meeting."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a quick update on a delayed project."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a congratulations on finishing a certification."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about feedback on a well-run sprint demo."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a heads-up about scheduled maintenance."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about an invitation to a lunch-and-learn."}]}
{"messages": [{"role": "user", "content": "Write an empathetic, professional message about a short status update for a stakeholder."}]}`;

// SFT demo: TOPIC CLASSIFICATION (ag_news-style). Each row is a prompt → single
// label (World / Sports / Business / Sci/Tech). A classic supervised task: the
// model imitates the right label, the eval scores label-accuracy, and the dataset
// is stratifiable (each class evenly represented). 60 rows, 15 per class.
export const DEMO_SFT_JSONL = `{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'UN council meets over escalating border tensions between neighboring states.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Peace talks resume as both sides agree to a temporary ceasefire.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Prime minister announces sweeping new immigration policy.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Floods displace thousands across the southern region.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Diplomats sign historic trade and security accord.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Election results spark mass protests in the capital.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Aid convoys reach the besieged city after weeks of delay.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Summit ends with pledge to cut global carbon emissions.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Embassy reopens a decade after diplomatic ties were severed.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Volcano eruption forces evacuation of coastal villages.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Parliament passes landmark human-rights legislation.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Refugee numbers climb as conflict spreads to the north.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Two nations restore air travel after long suspension.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Earthquake of magnitude 6.4 strikes the mountainous province.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Leaders gather for an emergency session on the refugee crisis.'"}, {"role": "assistant", "content": "World"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Home side wins the league title on the final day of the season.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Veteran striker scores hat-trick in derby victory.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Underdogs stun champions in overtime thriller.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Star quarterback signs record-breaking contract extension.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Olympic sprinter sets a new national record in the 200m.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Tennis seed advances to the semifinals after a five-set battle.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Coach resigns following a disappointing playoff exit.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Cyclist claims the yellow jersey in the mountain stage.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Rookie pitcher throws a no-hitter in just his third start.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'National team qualifies for the World Cup after a tense win.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Golfer sinks a long putt to force a playoff at the open.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Defending champions are eliminated in the quarterfinals.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Swimmer breaks the world record in the 100m freestyle.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Marathon record falls on a cool morning in the city.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Basketball team clinches a playoff berth with a buzzer-beater.'"}, {"role": "assistant", "content": "Sports"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Central bank raises interest rates to curb rising inflation.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Tech giant reports record quarterly profits, shares jump.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Oil prices slide on weaker-than-expected demand.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Startup raises 200 million dollars in its latest funding round.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Retail sales climb during the holiday shopping season.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Merger creates the largest airline in the region.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Unemployment falls to its lowest level in a decade.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Stock index closes at an all-time high amid earnings optimism.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Automaker recalls thousands of vehicles over a brake defect.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Currency weakens sharply after the trade-deficit report.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Quarterly earnings miss forecasts, sending shares lower.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Regulators approve the long-awaited banking merger.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Consumer confidence rebounds as gas prices ease.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Chipmaker forecasts strong demand into the next year.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Housing market cools as mortgage rates climb.'"}, {"role": "assistant", "content": "Business"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Researchers unveil a new battery that charges in minutes.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Astronomers detect water vapor on a distant exoplanet.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'New AI model sets a state-of-the-art benchmark for translation.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Spacecraft completes its first flyby of the asteroid belt.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Engineers demonstrate a quantum chip with record stability.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Biologists sequence the genome of a rare deep-sea species.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Smartphone maker debuts a foldable display prototype.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Telescope captures the sharpest image yet of a spiral galaxy.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Lab grows functional tissue using a 3D bioprinter.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Open-source project releases a faster web browser engine.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Rover finds evidence of ancient riverbeds on the red planet.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Scientists develop a vaccine candidate using mRNA technology.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Drone delivery trial begins in a suburban test market.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'Researchers report a breakthrough in solar-panel efficiency.'"}, {"role": "assistant", "content": "Sci/Tech"}]}
{"messages": [{"role": "user", "content": "Classify the topic of this news headline as World, Sports, Business, or Sci/Tech: 'New programming language gains traction among developers.'"}, {"role": "assistant", "content": "Sci/Tech"}]}`;

// Build a File object from bundled JSONL text so it can drop straight into the
// FileUpload control (same path as a user-picked file).
export function demoFile(kind: "sft" | "dpo" | "kto" | "rlvr" | "rlaif"): File {
  const text =
    kind === "sft" ? DEMO_SFT_JSONL
    : kind === "dpo" ? DEMO_DPO_JSONL
    : kind === "kto" ? DEMO_KTO_JSONL
    : kind === "rlvr" ? DEMO_RLVR_JSONL
    : DEMO_RLAIF_JSONL;
  const name =
    kind === "sft" ? "sft_news_topics_60.jsonl"
    : kind === "dpo" ? "dpo_support_120.jsonl"
    : kind === "kto" ? "kto_support_120.jsonl"
    : kind === "rlvr" ? "rlvr_gsm8k_100.jsonl"
    : "rlaif_notes_180.jsonl";
  return new File([text + "\n"], name, { type: "application/jsonl" });
}

// Trigger a browser download of the bundled demo file.
export function downloadDemo(kind: "sft" | "dpo" | "kto" | "rlvr" | "rlaif"): void {
  const f = demoFile(kind);
  const url = URL.createObjectURL(f);
  const a = document.createElement("a");
  a.href = url;
  a.download = f.name;
  a.click();
  URL.revokeObjectURL(url);
}
