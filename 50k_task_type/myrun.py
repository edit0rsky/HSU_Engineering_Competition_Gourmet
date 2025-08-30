import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import os
import json
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks


# === 메모리 안전 JSONL 로더 & 프리뷰 ===
def peek_jsonl(path, n=10):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    if not rows:
        print("[경고] 파일이 비어 있습니다:", path)
        return
    df_small = pd.DataFrame(rows)
    # 원래 코드의 .info() / .head(10) 출력 유지
    df_small.info()
    print(df_small.head(10))


def load_jsonl_efficient(path, emb_dim=3072):
    """
    JSONL을 스트리밍으로 읽어 메모리 안전하게 로드.
    반환: (user_ids_encoded, biz_ids_encoded, stars(float32), embeddings(float32 [N,emb_dim]), N)
    """
    # 1) 총 라인 수 파악
    with open(path, "r", encoding="utf-8") as f:
        N = sum(1 for _ in f)

    # 2) 컨테이너 준비
    users = [None] * N
    bizs = [None] * N
    stars = np.empty(N, dtype=np.int16)
    embs = np.empty((N, emb_dim), dtype=np.float32)

    # 3) 한 줄씩 파싱 → 곧바로 numpy에 채우기
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            o = json.loads(line)
            # 원래 스키마 그대로 사용
            users[i] = o["user_id"]
            bizs[i] = o["business_id"]
            # review_stars는 정수
            stars[i] = int(o["review_stars"])
            # embedding은 길이=emb_dim 가정
            embs[i] = o["embedding"]

    # 4) LabelEncoder 적용(원래 코드 동일)
    user_encoder = LabelEncoder()
    business_encoder = LabelEncoder()
    user_encoded = user_encoder.fit_transform(np.asarray(users, dtype=object)).astype(
        np.int32
    )
    biz_encoded = business_encoder.fit_transform(np.asarray(bizs, dtype=object)).astype(
        np.int32
    )

    num_users = len(user_encoder.classes_)
    num_businesses = len(business_encoder.classes_)

    # y는 float32로 변환(원래 코드에서 metrics 연산 float 기반)
    y_all = stars.astype(np.float32)

    return user_encoded, biz_encoded, y_all, embs, num_users, num_businesses, N


# ===============================
# ==== 사용자 원본 하이퍼파라미터/로직 그대로 ====
TASK = [
    "RETRIEVAL_QUERY",
    "CLASSIFICATION",
]

for task in TASK:
    input_file = (
        "/root/project/Gourmet-with-GeminiEmbedding/Dataset/3states/fl_split5_"
        + task
        + ".jsonl"
    )
    print(input_file)

    # 가벼운 프리뷰로 info/head 유지 (메모리 폭증 없이)
    peek_jsonl(input_file, n=10)

    # 전체 데이터셋 효율 로드 (DataFrame 대신 numpy)
    gemini_embedding_dim = 3072  # 원래 값 그대로
    user_all, biz_all, y_all, E_all, num_users, num_businesses, N = (
        load_jsonl_efficient(input_file, emb_dim=gemini_embedding_dim)
    )

    print(f"전체 데이터셋 크기: {N}")
    print(num_users)
    print(num_businesses)

    # ===== 원래 로직의 7:1:2 분할을 동일하게 재현 =====
    # 학습+검증 / 테스트 (test_size=0.2, random_state=42)
    idx_all = np.arange(N)
    train_val_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=42)

    # 학습 / 검증 (val_size_ratio = 1/8, random_state=42)  # 전체의 10% = 학습+검증의 12.5%
    val_size_ratio = 1 / 8
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_ratio, random_state=42
    )

    # 분할 결과 카운트 출력(원래 print 유지)
    print(f"전체 데이터 수: {N}")
    print(f"학습 데이터 수: {len(train_idx)} ({len(train_idx)/N*100:.2f}%)")
    print(f"검증 데이터 수: {len(val_idx)} ({len(val_idx)/N*100:.2f}%)")
    print(f"테스트 데이터 수: {len(test_idx)} ({len(test_idx)/N*100:.2f}%)")

    # === 원래 코드의 넘파이 배열 생성과 동일한 결과로 매핑 ===
    train_embeddings = E_all[train_idx]
    val_embeddings = E_all[val_idx]
    test_embeddings = E_all[test_idx]

    print(f"학습 임베딩 데이터 형태: {train_embeddings.shape}")
    print(f"검증 임베딩 데이터 형태: {val_embeddings.shape}")
    print(f"테스트 임베딩 데이터 형태: {test_embeddings.shape}")
    print(f"데이터 type: {train_embeddings.dtype}")

    # ====== 원래 모델/하이퍼파라미터 그대로 ======
    user_business_embedding_dim = 64
    user_biz_mlp_dims = [128, 64]
    final_mlp_dims = [32, 16]
    learning_rate = 0.001
    batch_size = 128

    # 입력층 정의 (원래 동일)
    user_input = keras.Input(shape=(1,), name="user_id_input")
    business_input = keras.Input(shape=(1,), name="business_id_input")
    user_embedding_layer = layers.Embedding(
        num_users, user_business_embedding_dim, name="user_embedding"
    )
    business_embedding_layer = layers.Embedding(
        num_businesses, user_business_embedding_dim, name="business_embedding"
    )
    user_vec = layers.Flatten()(user_embedding_layer(user_input))
    business_vec = layers.Flatten()(business_embedding_layer(business_input))
    combined_vec = layers.concatenate([user_vec, business_vec])
    interaction_features = combined_vec
    for dim in user_biz_mlp_dims:
        interaction_features = layers.Dense(dim, activation="relu")(
            interaction_features
        )

    gemini_input = keras.Input(
        shape=(gemini_embedding_dim,), name="gemini_embedding_input"
    )
    review_features = layers.Dense(1536, activation="relu")(gemini_input)
    review_features = layers.Dense(768, activation="relu")(review_features)
    review_features = layers.Dense(512, activation="relu")(review_features)

    final_combined_features = layers.concatenate(
        [interaction_features, review_features]
    )
    predicted_rating = final_combined_features
    for dim in final_mlp_dims:
        predicted_rating = layers.Dense(dim, activation="relu")(predicted_rating)
    output_rating = layers.Dense(1, activation="linear", name="output_rating")(
        predicted_rating
    )

    final_model = models.Model(
        inputs=[user_input, business_input, gemini_input], outputs=output_rating
    )

    # === 컴파일/콜백/에폭 등 그대로 ===
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae"],
    )

    final_model_base_path = f"final_best_gemini_model5_{task}"
    early_stopping_callback = callbacks.EarlyStopping(
        monitor="val_rmse",
        patience=10,
        min_delta=0.0005,
        mode="min",
        restore_best_weights=True,
    )
    final_model_path = f"{final_model_base_path}_main.keras"
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=final_model_path,
        monitor="val_rmse",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    print(f"\n==== [{task}] 버전 학습 시작")
    epochs = 50

    # === 여기서부터 입력은 DataFrame 컬럼 대신 우리가 만든 배열 사용 ===
    history = final_model.fit(
        {
            "user_id_input": user_all[train_idx],
            "business_id_input": biz_all[train_idx],
            "gemini_embedding_input": train_embeddings,
        },
        y_all[train_idx],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            {
                "user_id_input": user_all[val_idx],
                "business_id_input": biz_all[val_idx],
                "gemini_embedding_input": val_embeddings,
            },
            y_all[val_idx],
        ),
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        verbose=1,
    )

    # ===== 평가(원래 동일) =====
    final_model = keras.models.load_model(final_model_path)
    test_predictions = final_model.predict(
        {
            "user_id_input": user_all[test_idx],
            "business_id_input": biz_all[test_idx],
            "gemini_embedding_input": test_embeddings,
        }
    ).flatten()

    true_ratings = y_all[test_idx]
    mse = mean_squared_error(true_ratings, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_ratings, test_predictions)
    mape = mean_absolute_percentage_error(true_ratings, test_predictions) * 100

    print(f"{task}버전 모델 성능 평가")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # ===== 5회 반복(원래 동일, 콜백/체크포인트 run별 생성 유지) =====
    mse_scores, rmse_scores, mae_scores, mape_scores = [], [], [], []
    num_runs = 5
    for i in range(num_runs):
        print("\n" + "=" * 60)
        print(f"                   실험 {i+1}/{num_runs} 시작")
        print("=" * 60)

        user_input = keras.Input(shape=(1,), name="user_id_input")
        business_input = keras.Input(shape=(1,), name="business_id_input")
        gemini_input = keras.Input(
            shape=(gemini_embedding_dim,), name="gemini_embedding_input"
        )

        user_embedding_layer = layers.Embedding(
            num_users, user_business_embedding_dim, name="user_embedding"
        )
        business_embedding_layer = layers.Embedding(
            num_businesses, user_business_embedding_dim, name="business_embedding"
        )
        user_vec = layers.Flatten()(user_embedding_layer(user_input))
        business_vec = layers.Flatten()(business_embedding_layer(business_input))
        combined_vec = layers.concatenate([user_vec, business_vec])
        interaction_features = combined_vec
        for dim in user_biz_mlp_dims:
            interaction_features = layers.Dense(dim, activation="relu")(
                interaction_features
            )

        review_features = layers.Dense(1536, activation="relu")(gemini_input)
        review_features = layers.Dense(768, activation="relu")(review_features)
        review_features = layers.Dense(512, activation="relu")(review_features)

        final_combined_features = layers.concatenate(
            [interaction_features, review_features]
        )
        predicted_rating = final_combined_features
        for dim in final_mlp_dims:
            predicted_rating = layers.Dense(dim, activation="relu")(predicted_rating)
        output_rating = layers.Dense(1, activation="linear", name="output_rating")(
            predicted_rating
        )

        run_model = models.Model(
            inputs=[user_input, business_input, gemini_input], outputs=output_rating
        )
        run_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae"],
        )

        run_ckpt_path = f"{final_model_base_path}_run{i+1}.keras"
        run_es = callbacks.EarlyStopping(
            monitor="val_rmse",
            patience=10,
            min_delta=0.0005,
            mode="min",
            restore_best_weights=True,
        )
        run_ckpt = callbacks.ModelCheckpoint(
            filepath=run_ckpt_path,
            monitor="val_rmse",
            save_best_only=True,
            mode="min",
            verbose=0,
        )

        run_model.fit(
            {
                "user_id_input": user_all[train_idx],
                "business_id_input": biz_all[train_idx],
                "gemini_embedding_input": train_embeddings,
            },
            y_all[train_idx],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                {
                    "user_id_input": user_all[val_idx],
                    "business_id_input": biz_all[val_idx],
                    "gemini_embedding_input": val_embeddings,
                },
                y_all[val_idx],
            ),
            callbacks=[run_es, run_ckpt],
            verbose=0,
        )
        print(f"실험 {i+1}: 모델 학습 완료.")

        best_model = keras.models.load_model(run_ckpt_path)
        predictions = best_model.predict(
            {
                "user_id_input": user_all[test_idx],
                "business_id_input": biz_all[test_idx],
                "gemini_embedding_input": test_embeddings,
            }
        ).flatten()

        true_ratings = y_all[test_idx]
        mse = mean_squared_error(true_ratings, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_ratings, predictions)
        mape = mean_absolute_percentage_error(true_ratings, predictions) * 100

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)

        print(f"실험 {i+1} 결과 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

        # 메모리 정리(원래 패턴 유지)
        del run_model, best_model, predictions
        gc.collect()
        tf.keras.backend.clear_session()

    # 평균/표준편차 계산 및 출력(원래 그대로)
    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    avg_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    avg_mape = np.mean(mape_scores)
    std_mape = np.std(mape_scores)

    print("\n" + "=" * 60)
    print(f"{task}버전 모델 성능 통계 (5회 실행 평균)")
    print("=" * 60)
    print(f"평균 MSE: {avg_mse:.4f} (표준편차: {std_mse:.4f})")
    print(f"평균 RMSE: {avg_rmse:.4f} (표준편차: {std_rmse:.4f})")
    print(f"평균 MAE : {avg_mae:.4f} (표준편차: {std_mae:.4f})")
    print(f"평균 MAPE: {avg_mape:.2f}% (표준편차: {std_mape:.2f})")
    print("=" * 60)

    summary_data = {
        "Metric": ["MSE", "RMSE", "MAE", "MAPE (%)"],
        "Run 1": [mse_scores[0], rmse_scores[0], mae_scores[0], mape_scores[0]],
        "Run 2": [mse_scores[1], rmse_scores[1], mae_scores[1], mape_scores[1]],
        "Run 3": [mse_scores[2], rmse_scores[2], mae_scores[2], mape_scores[2]],
        "Run 4": [mse_scores[3], rmse_scores[3], mae_scores[3], mape_scores[3]],
        "Run 5": [mse_scores[4], rmse_scores[4], mae_scores[4], mape_scores[4]],
        "Average": [avg_mse, avg_rmse, avg_mae, avg_mape],
        "Std. Deviation": [std_mse, std_rmse, std_mae, std_mape],
    }

    results_df = pd.DataFrame(summary_data)
    results_df = results_df.round(
        {
            "Run 1": 4,
            "Run 2": 4,
            "Run 3": 4,
            "Run 4": 4,
            "Run 5": 4,
            "Average": 4,
            "Std. Deviation": 4,
        }
    )
    results_df.loc[results_df["Metric"] == "MAPE (%)"] = results_df.loc[
        results_df["Metric"] == "MAPE (%)"
    ].round(2)

    print(f"--- [{task}] 버전 최종 성능 요약 테이블 ---")
    results_df.to_csv(
        "model_performance5_" + task + ".csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("csv로 저장 완료")

    # 태스크 종료 시 그래프/세션 정리
    gc.collect()
    tf.keras.backend.clear_session()
