using Parquet, DataFrames
using MLJ, MLJModels
using Random
using Statistics
using TidierData

# get parameter from command line
args = collect(ARGS)
if length(args) == 0
    data_path = "./data/"
else
    data_path = args[1]
end

@show data_path

begin
    # Load the data
    local train_basetable = DataFrame(read_parquet(data_path * "parquet_files/train/train_base.parquet"))
    local train_static = vcat(
        DataFrame(read_parquet(data_path * "parquet_files/train/train_static_0_0.parquet")),
        DataFrame(read_parquet(data_path * "parquet_files/train/train_static_0_1.parquet"))
    )

    local train_static_cb = DataFrame(read_parquet(data_path * "parquet_files/train/train_static_cb_0.parquet"))
    local train_person_1 = DataFrame(read_parquet(data_path * "parquet_files/train/train_person_1.parquet"))

    local train_credit_bureau_b_2 = DataFrame(read_parquet(data_path * "parquet_files/train/train_credit_bureau_b_2.parquet"))

    local test_basetable = DataFrame(read_parquet(data_path * "parquet_files/test/test_base.parquet"))
    local test_static = vcat(
        DataFrame(read_parquet(data_path * "parquet_files/test/test_static_0_0.parquet")),
        DataFrame(read_parquet(data_path * "parquet_files/test/test_static_0_1.parquet")),
        DataFrame(read_parquet(data_path * "parquet_files/test/test_static_0_2.parquet"))
    )

    local test_static_cb = DataFrame(read_parquet(data_path * "parquet_files/test/test_static_cb_0.parquet"))
    local test_person_1 = DataFrame(read_parquet(data_path * "parquet_files/test/test_person_1.parquet"))

    local test_credit_bureau_b_2 = DataFrame(read_parquet(data_path * "parquet_files/test/test_credit_bureau_b_2.parquet"))

    # Feature engineering
    local train_person_1_feats_1 = combine(
        groupby(train_person_1, :case_id, sort=true),
        :mainoccupationinc_384A => maximum => :mainoccupationinc_384A_max,
        :incometype_1044T => x -> any(x .== "SELFEMPLOYED") => :mainoccupationinc_384A_any_selfemployed
    )

    local train_person_1_feats_2 = @chain train_person_1 begin
        @filter(num_group1 .== 0)
        select(:case_id, :housetype_905L)
        rename(:housetype_905L => :person_housetype)
    end

    local train_credit_bureau_b_2_feats = combine(
        groupby(train_credit_bureau_b_2, :case_id, sort=true),
        :pmts_pmtsoverdue_635A => maximum => :pmts_pmtsoverdue_635A_max,
        :pmts_dpdvalue_108P => x -> any(x .> 31) => :pmts_dpdvalue_108P_over31
    )

    local selected_static_cols = filter(x -> x[end] in ('A', 'M'), names(train_static))
    local selected_static_cb_cols = filter(x -> x[end] in ('A', 'M'), names(train_static_cb))

    local data = @chain train_basetable begin
        leftjoin(select(train_static, :case_id, selected_static_cols), on=:case_id)
        leftjoin(select(train_static_cb, :case_id, selected_static_cb_cols), on=:case_id)
        leftjoin(train_person_1_feats_1, on=:case_id)
        leftjoin(train_person_1_feats_2, on=:case_id)
        leftjoin(train_credit_bureau_b_2_feats, on=:case_id)
    end


    local test_person_1_feats_1 = combine(
        groupby(test_person_1, :case_id, sort=true),
        :mainoccupationinc_384A => maximum => :mainoccupationinc_384A_max,
        :incometype_1044T => x -> any(x .== "SELFEMPLOYED") => :mainoccupationinc_384A_any_selfemployed
    )

    local test_person_1_feats_2 = @chain test_person_1 begin
        @filter(num_group1 .== 0)
        select(:case_id, :housetype_905L)
        rename(:housetype_905L => :person_housetype)
    end

    local test_credit_bureau_b_2_feats = combine(
        groupby(test_credit_bureau_b_2, :case_id, sort=true),
        :pmts_pmtsoverdue_635A => maximum => :pmts_pmtsoverdue_635A_max,
        :pmts_dpdvalue_108P => x -> any(x .> 31) => :pmts_dpdvalue_108P_over31
    )

    global data_submission = @chain test_basetable begin
        leftjoin(select(test_static, :case_id, selected_static_cols), on=:case_id)
        leftjoin(select(test_static_cb, :case_id, selected_static_cb_cols), on=:case_id)
        leftjoin(test_person_1_feats_1, on=:case_id)
        leftjoin(test_person_1_feats_2, on=:case_id)
        leftjoin(test_credit_bureau_b_2_feats, on=:case_id)
    end

    local case_ids = Int64.(shuffle(unique(data.case_id)))
    local case_ids_train, case_ids_test = partition(case_ids, 0.6, shuffle=true)
    local case_ids_train, case_ids_test = Int64.(case_ids_train), Int64.(case_ids_test)

    local case_ids_valid, case_ids_test = partition(case_ids_test, 0.5, shuffle=true)
    local case_ids_valid, case_ids_test = Int64.(case_ids_valid), Int64.(case_ids_test)

    local cols_pred = filter(names(data)) do x
        x[1:end-1] == lowercase(x[1:end-1]) && isuppercase(x[end])
    end

    data = @chain data begin
        @select(case_id, WEEK_NUM, target, !!cols_pred...)
    end

    local schema = MLJ.schema(data)
    local cat_feats = [schema.names[[schema.scitypes...] .== Union{Missing, Textual}]...]

    coerce!(data, (cat_feats .=> Multiclass)...)

    local OneHotEncoder = @load OneHotEncoder pkg=MLJModels

    local imputer = FillImputer()
    local encoder = OneHotEncoder(features = cat_feats)

    local mach_hot = machine(encoder, data) |> fit!
    local mach_cont = machine(imputer, data) |> fit!

    function split_and_filter(case_ids) # the same as from_polars_to_pandas in Python code
        base = @chain data begin
            @filter(case_id in !!case_ids)
            @select(case_id, WEEK_NUM, target)
        end

        X = @chain data begin
            @filter(case_id in !!case_ids)
            @select(!!cols_pred...)
        end

        y = @chain data begin
            @filter(case_id in !!case_ids)
            @select(target)
        end

        return base, X, y
    end

    global base_train, X_train, y_train = split_and_filter(case_ids_train)
    global base_valid, X_valid, y_valid = split_and_filter(case_ids_valid)
    global base_test, X_test, y_test = split_and_filter(case_ids_test)

    coerce!(X_train, (cat_feats .=> Multiclass)...)
    coerce!(X_valid, (cat_feats .=> Multiclass)...)
    coerce!(X_test, (cat_feats .=> Multiclass)...)

    coerce!(y_train, (cat_feats .=> Multiclass)...)
    coerce!(y_valid, (cat_feats .=> Multiclass)...)
    coerce!(y_test, (cat_feats .=> Multiclass)...)

    X_train = MLJ.transform(mach_hot, MLJ.transform(mach_cont, X_train))
    X_valid = MLJ.transform(mach_hot, MLJ.transform(mach_cont, X_valid))
    X_test = MLJ.transform(mach_hot, MLJ.transform(mach_cont, X_test))

    # should only change types to non-missing
    dropmissing!(X_train)
    dropmissing!(X_valid)
    dropmissing!(X_test)
    dropmissing!(y_train)
    dropmissing!(y_valid)
    dropmissing!(y_test)

    @show size(X_train)
    @show size(X_valid)
    @show size(X_test)
end;

DecisionTreeClassifier = @load DecisionTreeClassifier pkg = "DecisionTree"

model = DecisionTreeClassifier()

mach = machine(model, X_train, y_train)

MLJ.fit!(mach, verbosity=1, force=true)

cv=CV(nfolds=3)

@show evaluate!(mach; measure=auc, verbosity=2, check_measure=false)
