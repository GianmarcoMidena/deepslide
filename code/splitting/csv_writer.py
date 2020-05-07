from img2ds import writing


class CSVWriter(writing.CSVWriter):
    def _write_example(self, path, label):
        self._data = self._data.append({'id': str(path.stem), 'path': path, 'label': label},
                                       ignore_index=True, sort=False)
