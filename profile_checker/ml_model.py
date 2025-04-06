import joblib
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# ✅ Expanded Dataset (100 Fake, 100 Real)
fake_usernames = [
    # ✅ Obvious Fake Profiles
    "user12345", "bot_account99", "fake_acc1001", "trial_acc", "free_promo2023",
    "clickhere_earn", "giveaway_1122", "insta_fan_333", "auto_generated_675", "new_acc_2024",
    "follow_me_now", "win_cash_$$$", "likes_boost_500", "fake_fake_fake", "scam_alert_999",
    "random_user_x", "bot_9988", "fake_profile_please", "follow4follow_777", "account_test_123",
    "test_account9876", "no_real_name_123", "xx_fakeuser_xx", "bot_like_this", "ai_generated_bot",
    "user09876_test", "hacker_pro_007", "xyz_bot_22", "buy_followers_2024", "click_now_spam",
    "randomized_test_account", "trial_user_fake", "giveaway_bot_101", "spammer_user1999",
    "not_a_real_person_543", "test_user_1123", "bot9999_xoxo", "fakeidentity_creator",
    "unknown_user_888", "profile_test_909", "win_money_now!", "fake_name_alert",
    "no_profile_pic_2025", "testing_bot_567", "lorem_ipsum_user", "new_fake_909",
    "xxx_randomxxx", "verify_now_urgent", "banned_account_001", "fake_detected_bot",
    "xoxo_bot_fake", "unknown_identity_777", "password_sharer_500", "gamer_bot_2024",
    
    # ✅ Less Obvious Fake Profiles
    "alex_human1990", "michael.k1", "real_jessica00", "sammy2001", "peter.p_99",
    "laura_new_93", "sophia.lee_test", "not_a_bot_77", "username_hidden_999", "the_real_dave56",
    "matt_jones_yt", "codingguy_101", "geeky_gamer2020", "newbie_techx", "unknown_human_82",
    "just.a.name_x", "try_this_name_2025", "sportsfan_2004", "pro_gamer_xo", "not_a_spammer_007",
    "legit_username2023", "music_fanatic_88", "invisible_id_400", "mynameisnotbot_009",
    "profile_user_actual", "mistaken_identity_1", "no_fake_here_2001", "idontcheat_123",
    "user_not_found_75", "valid_account12", "safe_profile_xo", "verified_user_legit",
    "real_deal_900", "social_guru_2025", "not_random_guy", "username_protected",
    "confidential_acc", "not_a_test", "serious_guy_here", "follower_legit_2024",
]

real_usernames = [
    "alexander_smith", "michael_jordan", "jessica.smith", "davidbrown12", "sophia_lee_33",
    "richard_adams", "emily_clark", "harry_wilson", "johnson_mike22", "samantha_d",
    "daniel_roberts99", "laura.perez", "the_real_ben", "jake_taylor_official", "hannah.green",
    "natalie_james_", "mark_andrews21", "patrick_hall", "elizabeth_owen_", "robert.carter",
    "kevin_martinez", "nicole.harris88", "paul_watson_1985", "diana_flores", "christopher_king",
    "mary_sullivan_", "charles_allen_", "lisa_gonzalez", "matthew_baker", "michelle_cooper",
    "brian_morris_", "angela_reed_", "timothy_cook_", "rachel_jackson_", "steven_thomas",
    "brandon_white_", "ashley_hall_", "scott_howard_", "jason_turner_", "megan_wright",
    "andrew_scott_", "kimberly_lopez", "ryan_edwards_", "thomas_hill_", "gregory_green",
    "linda_baker_", "christina_watson_", "jeffrey_mitchell", "cheryl_foster_", "victoria_carter",
    "john_thomas_", "margaret_clark_", "jeremy_patterson", "patrick_collins_", "stephanie_harris",
    "dylan_roberts_", "robin_wilson_", "eric_martinez_", "joseph_jackson_", "katherine_anderson",
    "melanie_wright_", "donald_turner_", "rebecca_walker_", "william_miller_", "brenda_evans_",
]

# Labels: 1 = Fake, 0 = Real
X = fake_usernames + real_usernames
y = [1] * len(fake_usernames) + [0] * len(real_usernames)

# ✅ Advanced feature extraction: Bi-grams to four-grams
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))  # Using bi-grams to four-grams
X_vectorized = vectorizer.fit_transform(X)

# ✅ Train the model
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# ✅ Ensure "profile_checker/" directory exists
save_dir = "D:/Profile Checking/fake_profile_detection/profile_checker/"
os.makedirs(save_dir, exist_ok=True)

# ✅ Save model & vectorizer
joblib.dump(model, os.path.join(save_dir, "svm_model.pkl"))
joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.pkl"))

print("✅ Model Trained and Saved Successfully at", save_dir)