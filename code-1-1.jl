struct State
    row
    column
end

function clone(model::State)
    return State(model.row, model.column)
end

@enum Action UP=1 DOWN=-1 LEFT=2 RIGHT=-2
