function allBookData = LoadAll(fileList)
  allBookData = [];
  for i=1:length(fileList)
    bookData = LoadBatch(fileList{i});
    allBookData = [allBookData bookData];
  endfor
endfunction
