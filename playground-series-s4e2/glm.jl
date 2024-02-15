using CSV: read, write
using DataFrames
using TidierData
using GLM

test_df = read("test.csv", DataFrame);
train_df = read("train.csv", DataFrame);

levels = Dict(
	"Insufficient_Weight" => -1,
	"Normal_Weight" => 0,
	"Overweight_Level_I" => 1,
	"Overweight_Level_II" => 2,
	"Obesity_Type_I" => 3,
	"Obesity_Type_II" => 4,
	"Obesity_Type_III" => 5,
    -1 => "Insufficient_Weight",
    0 => "Normal_Weight",
    1 => "Overweight_Level_I",
    2 => "Overweight_Level_II",
    3 => "Obesity_Type_I",
    4 => "Obesity_Type_II",
    5 => "Obesity_Type_III"
)

train_df[!, :Y] = map(x -> levels[x], train_df[!, :NObeyesdad])

@glimpse(train_df)

ols = lm(@formula(Y ~ Gender + Age + Height + Weight + family_history_with_overweight + FAVC + NCP + CAEC + SMOKE + SCC + FAF + MTRANS), train_df)

test_df.NObeyesdad = clamp.(Int.(round.(predict(ols, test_df))), -1, 5) .|> x -> levels[x]

write("submission.csv", select(test_df, :id, :NObeyesdad))

