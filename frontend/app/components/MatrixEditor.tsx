"use client";

type MatrixEditorProps = {
  matrix: number[][];
  onChange: (matrix: number[][]) => void;
};

export default function MatrixEditor({
  matrix,
  onChange,
}: MatrixEditorProps) {
  const updateCell = (
    rowIndex: number,
    columnIndex: number,
    value: string,
  ) => {
    const nextMatrix = matrix.map((row) => [...row]);

    const parsedValue = Number(value);

    nextMatrix[rowIndex][columnIndex] =
      value === "" || Number.isNaN(parsedValue) ? 0 : parsedValue;

    onChange(nextMatrix);
  };

  const addRow = () => {
    const columns = matrix[0]?.length ?? 1;

    onChange([
      ...matrix,
      Array.from({ length: columns }, () => 0),
    ]);
  };

  const removeRow = () => {
    if (matrix.length <= 1) return;

    onChange(matrix.slice(0, -1));
  };

  const addColumn = () => {
    onChange(matrix.map((row) => [...row, 0]));
  };

  const removeColumn = () => {
    if ((matrix[0]?.length ?? 0) <= 1) return;

    onChange(matrix.map((row) => row.slice(0, -1)));
  };

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded-xl border border-zinc-200 bg-white p-4">
        <table className="border-collapse">
          <tbody>
            {matrix.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((value, columnIndex) => (
                  <td key={columnIndex} className="p-1">
                    <input
                      type="number"
                      value={value}
                      onChange={(event) =>
                        updateCell(
                          rowIndex,
                          columnIndex,
                          event.target.value,
                        )
                      }
                      className="h-12 w-20 rounded-lg border border-zinc-300 bg-white px-3 text-center text-sm font-medium text-zinc-900 outline-none transition focus:border-zinc-900 focus:ring-2 focus:ring-zinc-200"
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={addRow}
          className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100"
        >
          + Row
        </button>

        <button
          type="button"
          onClick={removeRow}
          className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100"
        >
          − Row
        </button>

        <button
          type="button"
          onClick={addColumn}
          className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100"
        >
          + Column
        </button>

        <button
          type="button"
          onClick={removeColumn}
          className="rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium text-zinc-700 transition hover:bg-zinc-100"
        >
          − Column
        </button>
      </div>
    </div>
  );
}
