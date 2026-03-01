function doPost(e) {
  var lock = LockService.getScriptLock();
  
  try {
    // 1. Validate Input
    if (!e || !e.postData || !e.postData.contents) {
      return responseJSON({ "status": "error", "message": "No data received" });
    }

    var payload = JSON.parse(e.postData.contents);

    // --- SCENARIO 1: PYTHON SCRIPT (Data Upload) ---
    // Now intercepts the "update_all" action from the new watcher.py
    if (payload.action === "update_all" || payload.fridge) {
      if (lock.tryLock(10000)) { // Wait up to 10s
        try {
          if (!authenticateUser(payload.username, payload.password)) {
            return responseJSON({ "status": "error", "message": "Invalid Credentials" });
          }
          return updateAllInventories(payload.fridge, payload.weekly, payload.monthly);
        } finally {
          lock.releaseLock();
        }
      } else {
        return responseJSON({ "status": "error", "message": "Server busy, try again" });
      }
    }

    // --- SCENARIO 2: FRONTEND ACTIONS ---
    var action = payload.action;

    // A. Read Actions (No Lock Needed - Faster)
    if (action === "login") {
      if (authenticateUser(payload.username, payload.password)) {
        return responseJSON({ "status": "success", "message": "Login Successful" });
      } else {
        return responseJSON({ "status": "error", "message": "Invalid Username or Password" });
      }
    }

    if (action === "get_inventory") {
      if (authenticateUser(payload.username, payload.password)) {
        return getInventoryData();
      } else {
        return responseJSON({ "error": "Invalid Credentials" });
      }
    }

    // B. Write Actions (Lock Needed)
    if (action === "signup" || action === "change_password") {
      if (lock.tryLock(10000)) {
        try {
          if (action === "signup") {
            return registerUser(payload.username, payload.password);
          }
          if (action === "change_password") {
            return changePassword(payload.username, payload.old_password, payload.new_password);
          }
        } finally {
          lock.releaseLock();
        }
      } else {
        return responseJSON({ "status": "error", "message": "Server busy" });
      }
    }

    return responseJSON({ "status": "error", "message": "Unknown Action" });

  } catch (err) {
    return responseJSON({ "status": "error", "message": "Server Error: " + err.toString() });
  }
}

// --- HELPER FUNCTIONS ---

function updateAllInventories(fridgeItems, weeklyItems, monthlyItems) {
  // Routes the three different JSON lists to their correct sheet tabs
  updateSheetData("Sheet1", fridgeItems);
  updateSheetData("Sheet Weekly", weeklyItems);
  updateSheetData("Sheet Monthly", monthlyItems);
  
  return responseJSON({ "status": "success", "message": "All Inventories Updated" });
}

function updateSheetData(sheetName, newItems) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(sheetName);
  if (!sheet) return; // Fails gracefully if you haven't created the sheet yet
  
  sheet.clearContents();
  
  if (!newItems || newItems.length === 0) return;

  var headers = Object.keys(newItems[0]);
  var values = [headers];
  for (var i = 0; i < newItems.length; i++) {
    var row = [];
    for (var h = 0; h < headers.length; h++) {
      row.push(newItems[i][headers[h]] || "");
    }
    values.push(row);
  }
  sheet.getRange(1, 1, values.length, values[0].length).setValues(values);
}

function getInventoryData() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Sheet1");
  var data = sheet.getDataRange().getValues();
  
  if (data.length < 2) return responseJSON({ "data": [] }); 

  var headers = data[0];
  var rows = data.slice(1);
  var result = [];

  for (var i = 0; i < rows.length; i++) {
    var obj = {};
    for (var j = 0; j < headers.length; j++) {
      obj[headers[j]] = rows[i][j];
    }
    result.push(obj);
  }

  return responseJSON({ "data": result });
}

function authenticateUser(user, pass) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Users");
  if (!sheet) return false;
  var data = sheet.getDataRange().getValues();
  for (var i = 1; i < data.length; i++) {
    if (String(data[i][0]) === String(user) && String(data[i][1]) === String(pass)) {
      return true;
    }
  }
  return false;
}

function registerUser(user, pass) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Users");
  var data = sheet.getDataRange().getValues();
  
  for (var i = 1; i < data.length; i++) {
    if (String(data[i][0]) === String(user)) {
      return responseJSON({ "status": "error", "message": "Username already exists" });
    }
  }
  
  sheet.appendRow([user, pass]);
  return responseJSON({ "status": "success", "message": "User Registered" });
}

function changePassword(user, oldPass, newPass) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Users");
  var data = sheet.getDataRange().getValues();
  
  for (var i = 1; i < data.length; i++) {
    if (String(data[i][0]) === String(user) && String(data[i][1]) === String(oldPass)) {
      sheet.getRange(i + 1, 2).setValue(newPass);
      return responseJSON({ "status": "success", "message": "Password Updated" });
    }
  }
  return responseJSON({ "status": "error", "message": "Old password incorrect" });
}

function responseJSON(data) {
  return ContentService.createTextOutput(JSON.stringify(data)).setMimeType(ContentService.MimeType.JSON);
}
