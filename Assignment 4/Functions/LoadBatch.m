function bookData = LoadBatch(fileName)
  % This function acts as a trivial .txt-load interface
  % It takes in a filename, opens it as long as it's stored in /Datasets,
  % dumps the data to a variable
  % closes the file and returns the data.
  %
  % INPUT:
  %   fileName -- string variable containing the full filename
  %
  % OUTPUT:
  %   bookData -- A vector containing the entirety of the file
  % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
  warning('off', 'all'); %Shuts down the filepath warnings
  addpath Datasets/; %Adds the standard path for files
  fid = fopen(fileName, 'r'); %Opens the file into fid
  bookData = fscanf(fid, '%c'); %scans the entire file and dumps it to bookData
  fclose(fid); %Closes the file
endfunction %ends the function
