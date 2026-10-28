# Five-arm information completion — exploratory

Records 3840/3840; reused 1024; new 2816.

| Old stratum | n | Native success | Shared success | Native C | Shared C | C reduction |
|---|---:|---:|---:|---:|---:|---:|
| all | 24 | 0.2677 [0.1656, 0.3750] | 0.4120 [0.2661, 0.5589] | 0.0190 [0.0075, 0.0328] | 0.0115 [-0.0001, 0.0285] | 0.0076 [-0.0029, 0.0188] |
| Dead | 8 | 0.0016 [0.0000, 0.0047] | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | -0.0000 [-0.0000, 0.0000] |
| Indifferent | 8 | 0.4062 [0.2422, 0.5875] | 0.4703 [0.2953, 0.6609] | 0.0060 [0.0000, 0.0137] | 0.0000 [-0.0021, 0.0025] | 0.0059 [-0.0005, 0.0140] |
| Fork | 8 | 0.3953 [0.2609, 0.5172] | 0.7656 [0.6063, 0.8969] | 0.0511 [0.0248, 0.0771] | 0.0344 [0.0005, 0.0788] | 0.0168 [-0.0140, 0.0471] |

C1-C0 = 2 Cov(q0, lift) + Var(lift). Same-repeat noise is removed.
Frozen historical labels are unchanged. Profiles are descriptive, not proof of zero effect.

## Information allocation (same collected cells)

| Scenario | Mean success | C |
|---|---:|---:|
| native | 0.2677 [0.1656, 0.3750] | 0.0190 [0.0075, 0.0328] |
| lottery20 | 0.2966 [0.1863, 0.4086] | 0.0147 [0.0050, 0.0262] |
| selective | 0.3073 [0.1948, 0.4214] | 0.0282 [0.0108, 0.0490] |
| broadcast | 0.4120 [0.2661, 0.5589] | 0.0115 [-0.0001, 0.0285] |

Selective: clue only on the metadata-fixed arm. Lottery: clue with probability 0.2 independently of action. Both have expected injection frequency 1/5 under uniform arms.
These are derived intervention protocols; no additional independent rollouts were run.

## Same-budget profiles (descriptive)

{
  "broadcast": {
    "Detected-Fork \u2192 Live-no-detected-Fork": 5,
    "Constant-failure \u2192 Constant-failure": 7,
    "Detected-Fork \u2192 Detected-Fork": 2,
    "Live-no-detected-Fork \u2192 Live-no-detected-Fork": 8,
    "Detected-Fork \u2192 Constant-success": 1,
    "Live-no-detected-Fork \u2192 Constant-failure": 1
  },
  "selective": {
    "Detected-Fork \u2192 Detected-Fork": 8,
    "Constant-failure \u2192 Constant-failure": 7,
    "Live-no-detected-Fork \u2192 Detected-Fork": 2,
    "Live-no-detected-Fork \u2192 Live-no-detected-Fork": 7
  }
}

## Every arm

| Point | Old regime | Arm | Action | Native q | Shared q |
|---|---|---|---|---:|---:|
| s029_t12 | Fork | a0 | examine desk 1 | 0.625 | 0.938 |
| s029_t12 | Fork | a1 | go to sidetable 2 | 0.875 | 0.875 |
| s029_t12 | Fork | a2 | go to sidetable 1 | 0.625 | 0.938 |
| s029_t12 | Fork | a3 | go to bed 1 | 0.375 | 1.000 |
| s029_t12 | Fork | a4 | look | 0.438 | 0.938 |
| s005_t31 | Dead | a0 | examine desk 1 | 0.000 | 0.000 |
| s005_t31 | Dead | a1 | go to sidetable 1 | 0.000 | 0.000 |
| s005_t31 | Dead | a2 | go to drawer 1 | 0.000 | 0.000 |
| s005_t31 | Dead | a3 | go to cabinet 1 | 0.000 | 0.000 |
| s005_t31 | Dead | a4 | inventory | 0.000 | 0.000 |
| s044_t18 | Fork | a0 | go to countertop 1 | 0.812 | 0.938 |
| s044_t18 | Fork | a1 | go to cabinet 4 | 0.062 | 0.688 |
| s044_t18 | Fork | a2 | go to fridge 1 | 0.250 | 0.875 |
| s044_t18 | Fork | a3 | examine cabinet 2 | 0.188 | 0.812 |
| s044_t18 | Fork | a4 | go to cabinet 1 | 0.125 | 0.875 |
| s007_t25 | Dead | a0 | examine sidetable 1 | 0.000 | 0.000 |
| s007_t25 | Dead | a1 | go to drawer 1 | 0.000 | 0.000 |
| s007_t25 | Dead | a2 | go to sidetable 2 | 0.000 | 0.000 |
| s007_t25 | Dead | a3 | go to sidetable 3 | 0.000 | 0.000 |
| s007_t25 | Dead | a4 | examine pen 1 | 0.000 | 0.000 |
| s010_t29 | Fork | a0 | go to drawer 1 | 0.312 | 0.938 |
| s010_t29 | Fork | a1 | examine dresser 1 | 0.250 | 1.000 |
| s010_t29 | Fork | a2 | look | 0.188 | 1.000 |
| s010_t29 | Fork | a3 | take pencil 1 from dresser 1 | 0.000 | 0.000 |
| s010_t29 | Fork | a4 | go to shelf 1 | 1.000 | 1.000 |
| s020_t01 | Indifferent | a0 | go to coffeetable 1 | 0.062 | 0.188 |
| s020_t01 | Indifferent | a1 | go to drawer 1 | 0.000 | 0.125 |
| s020_t01 | Indifferent | a2 | go to sidetable 1 | 0.000 | 0.062 |
| s020_t01 | Indifferent | a3 | go to armchair 1 | 0.000 | 0.062 |
| s020_t01 | Indifferent | a4 | go to diningtable 1 | 0.188 | 0.375 |
| s051_t08 | Indifferent | a0 | go to countertop 1 | 0.188 | 0.375 |
| s051_t08 | Indifferent | a1 | go to cabinet 1 | 0.312 | 0.188 |
| s051_t08 | Indifferent | a2 | examine cabinet 4 | 0.375 | 0.062 |
| s051_t08 | Indifferent | a3 | go to fridge 1 | 0.125 | 0.188 |
| s051_t08 | Indifferent | a4 | go to cabinet 5 | 0.375 | 0.312 |
| s006_t24 | Dead | a0 | examine desk 1 | 0.000 | 0.000 |
| s006_t24 | Dead | a1 | look | 0.000 | 0.000 |
| s006_t24 | Dead | a2 | inventory | 0.000 | 0.000 |
| s006_t24 | Dead | a3 | go to drawer 1 | 0.000 | 0.000 |
| s006_t24 | Dead | a4 | go to shelf 1 | 0.000 | 0.000 |
| s003_t18 | Dead | a0 | go to countertop 1 | 0.000 | 0.000 |
| s003_t18 | Dead | a1 | go to cabinet 4 | 0.000 | 0.000 |
| s003_t18 | Dead | a2 | go to cabinet 1 | 0.000 | 0.000 |
| s003_t18 | Dead | a3 | go to drawer 1 | 0.000 | 0.000 |
| s003_t18 | Dead | a4 | go to stoveburner 1 | 0.000 | 0.000 |
| s095_t01 | Dead | a0 | go to diningtable 1 | 0.000 | 0.000 |
| s095_t01 | Dead | a1 | go to countertop 1 | 0.000 | 0.000 |
| s095_t01 | Dead | a2 | go to diningtable 2 | 0.000 | 0.000 |
| s095_t01 | Dead | a3 | go to cabinet 1 | 0.000 | 0.000 |
| s095_t01 | Dead | a4 | look | 0.000 | 0.000 |
| s076_t16 | Fork | a0 | go to drawer 1 | 0.438 | 1.000 |
| s076_t16 | Fork | a1 | go to armchair 1 | 1.000 | 1.000 |
| s076_t16 | Fork | a2 | go to sidetable 1 | 0.438 | 1.000 |
| s076_t16 | Fork | a3 | go to shelf 1 | 0.312 | 1.000 |
| s076_t16 | Fork | a4 | examine dresser 2 | 0.188 | 1.000 |
| s087_t19 | Indifferent | a0 | go to cabinet 4 | 0.375 | 0.500 |
| s087_t19 | Indifferent | a1 | examine cabinet 3 | 0.438 | 0.375 |
| s087_t19 | Indifferent | a2 | go to drawer 1 | 0.375 | 0.250 |
| s087_t19 | Indifferent | a3 | go to countertop 1 | 0.500 | 0.312 |
| s087_t19 | Indifferent | a4 | go to cabinet 1 | 0.250 | 0.500 |
| s025_t31 | Indifferent | a0 | cool potato 1 with fridge 1 | 0.312 | 0.375 |
| s025_t31 | Indifferent | a1 | examine fridge 1 | 0.125 | 0.438 |
| s025_t31 | Indifferent | a2 | go to countertop 1 | 0.438 | 0.250 |
| s025_t31 | Indifferent | a3 | go to countertop 2 | 0.438 | 0.188 |
| s025_t31 | Indifferent | a4 | examine potato 1 | 0.500 | 0.188 |
| s097_t08 | Indifferent | a0 | go to toilet 1 | 1.000 | 0.938 |
| s097_t08 | Indifferent | a1 | examine cabinet 3 | 1.000 | 0.938 |
| s097_t08 | Indifferent | a2 | go to toiletpaperhanger 1 | 0.875 | 1.000 |
| s097_t08 | Indifferent | a3 | open cabinet 3 | 0.875 | 0.938 |
| s097_t08 | Indifferent | a4 | go to cabinet 2 | 0.938 | 0.938 |
| s094_t07 | Indifferent | a0 | go to cabinet 2 | 0.062 | 0.250 |
| s094_t07 | Indifferent | a1 | go to fridge 1 | 0.250 | 0.438 |
| s094_t07 | Indifferent | a2 | examine cabinet 1 | 0.188 | 0.250 |
| s094_t07 | Indifferent | a3 | open cabinet 1 | 0.062 | 0.438 |
| s094_t07 | Indifferent | a4 | go to diningtable 1 | 0.312 | 0.250 |
| s034_t13 | Indifferent | a0 | go to bed 1 | 0.750 | 0.625 |
| s034_t13 | Indifferent | a1 | examine drawer 1 | 0.438 | 0.812 |
| s034_t13 | Indifferent | a2 | go to desk 1 | 0.500 | 0.812 |
| s034_t13 | Indifferent | a3 | go to drawer 2 | 0.375 | 0.750 |
| s034_t13 | Indifferent | a4 | close drawer 1 | 0.312 | 0.625 |
| s069_t09 | Indifferent | a0 | open cabinet 3 | 0.500 | 0.562 |
| s069_t09 | Indifferent | a1 | examine cabinet 3 | 0.375 | 0.625 |
| s069_t09 | Indifferent | a2 | go to cabinet 1 | 0.688 | 0.750 |
| s069_t09 | Indifferent | a3 | go to cabinet 4 | 0.438 | 0.688 |
| s069_t09 | Indifferent | a4 | go to diningtable 1 | 0.938 | 0.875 |
| s077_t42 | Fork | a0 | examine countertop 2 | 0.000 | 0.312 |
| s077_t42 | Fork | a1 | go to countertop 4 | 0.188 | 0.250 |
| s077_t42 | Fork | a2 | go to cabinet 1 | 0.000 | 0.312 |
| s077_t42 | Fork | a3 | examine coffeemachine 1 | 0.000 | 0.375 |
| s077_t42 | Fork | a4 | go to countertop 1 | 0.000 | 0.125 |
| s021_t15 | Dead | a0 | go to countertop 1 | 0.000 | 0.000 |
| s021_t15 | Dead | a1 | go to fridge 1 | 0.000 | 0.000 |
| s021_t15 | Dead | a2 | examine drawer 3 | 0.000 | 0.000 |
| s021_t15 | Dead | a3 | examine drawer 2 | 0.062 | 0.000 |
| s021_t15 | Dead | a4 | open drawer 3 | 0.000 | 0.000 |
| s015_t05 | Dead | a0 | go to countertop 2 | 0.000 | 0.000 |
| s015_t05 | Dead | a1 | go to countertop 1 | 0.000 | 0.000 |
| s015_t05 | Dead | a2 | examine cabinet 2 | 0.000 | 0.000 |
| s015_t05 | Dead | a3 | go to cabinet 3 | 0.000 | 0.000 |
| s015_t05 | Dead | a4 | examine cabinet 1 | 0.000 | 0.000 |
| s023_t27 | Fork | a0 | examine countertop 1 | 0.188 | 1.000 |
| s023_t27 | Fork | a1 | go to cabinet 4 | 0.062 | 0.938 |
| s023_t27 | Fork | a2 | go to countertop 2 | 0.812 | 0.812 |
| s023_t27 | Fork | a3 | go to cabinet 1 | 0.125 | 0.812 |
| s023_t27 | Fork | a4 | go to cabinet 2 | 0.125 | 1.000 |
| s019_t09 | Fork | a0 | examine cabinet 3 | 0.562 | 0.562 |
| s019_t09 | Fork | a1 | open cabinet 3 | 0.688 | 0.750 |
| s019_t09 | Fork | a2 | go to cabinet 1 | 0.562 | 0.625 |
| s019_t09 | Fork | a3 | go to countertop 1 | 0.625 | 0.938 |
| s019_t09 | Fork | a4 | go to cabinet 4 | 0.500 | 0.625 |
| s078_t36 | Dead | a0 | examine countertop 2 | 0.000 | 0.000 |
| s078_t36 | Dead | a1 | go to countertop 3 | 0.000 | 0.000 |
| s078_t36 | Dead | a2 | go to countertop 1 | 0.000 | 0.000 |
| s078_t36 | Dead | a3 | go to fridge 1 | 0.000 | 0.000 |
| s078_t36 | Dead | a4 | go to cabinet 1 | 0.000 | 0.000 |
| s073_t10 | Fork | a0 | go to countertop 1 | 0.938 | 0.812 |
| s073_t10 | Fork | a1 | go to cabinet 1 | 0.750 | 0.938 |
| s073_t10 | Fork | a2 | go to drawer 1 | 0.562 | 0.812 |
| s073_t10 | Fork | a3 | take soapbottle 1 from cabinet 3 | 0.062 | 0.000 |
| s073_t10 | Fork | a4 | go to sinkbasin 1 | 0.562 | 0.812 |
